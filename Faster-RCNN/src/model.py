import torch
import torch.nn as nn
import torchvision
from torchvision.ops import box_iou, RoIPool
from utils import apply_deltas

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# Backbone (VGG16 w/o last maxpool)
# -----------------------
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:-1])

    def forward(self, x):
        return self.backbone(x)

# -----------------------
# Region Proposal Network
# -----------------------
class RPN(nn.Module):
    def __init__(self, in_channel=512):
        super(RPN, self).__init__()
        self.scales = [4, 8, 16, 32, 64, 128, 256, 512]
        self.aspect_ratios = [0.25, 0.5, 1.0, 2.0, 2.5]
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)

        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(in_channel, self.num_anchors, kernel_size=1)
        self.box_reg_layer = nn.Conv2d(in_channel, self.num_anchors * 4, kernel_size=1)

    def generate_anchors(self, image, feature_map):
        grid_h, grid_w = feature_map.shape[-2:]
        image_h, image_w = image.shape[-2:]
        stride_h = image_h // grid_h
        stride_w = image_w // grid_w

        scales = torch.tensor(self.scales, dtype=torch.float32, device=feature_map.device)
        ratios = torch.tensor(self.aspect_ratios, dtype=torch.float32, device=feature_map.device)
        h_ratios = torch.sqrt(ratios)
        w_ratios = 1.0 / h_ratios
        ws = (w_ratios[:, None] * scales[None, :]).reshape(-1)
        hs = (h_ratios[:, None] * scales[None, :]).reshape(-1)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2.0

        shift_x = torch.arange(grid_w, device=feature_map.device) * stride_w
        shift_y = torch.arange(grid_h, device=feature_map.device) * stride_h
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4)

        return (shifts[:, None, :] + base_anchors[None, :, :]).reshape(-1, 4)

    def clamp_boxes(self, boxes, image_shape):
        h, w = image_shape[-2:]
        x1 = boxes[..., 0].clamp(min=0, max=w)
        y1 = boxes[..., 1].clamp(min=0, max=h)
        x2 = boxes[..., 2].clamp(min=0, max=w)
        y2 = boxes[..., 3].clamp(min=0, max=h)
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def filter_proposals(self, proposals, cls_scores, image_shape):
        cls_scores = torch.sigmoid(cls_scores.view(-1))
        topk = min(10000, cls_scores.shape[0])
        scores, idx = cls_scores.topk(topk)
        proposals = self.clamp_boxes(proposals[idx], image_shape)
        scores = scores[:len(proposals)]
        keep = torchvision.ops.nms(proposals, scores, iou_threshold=0.7)
        return proposals[keep][:2000], scores[keep][:2000]

    def box_coder_encode(self, anchors, gt_boxes):
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_w
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_h

        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_w
        gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_h

        dx = (gt_ctr_x - anchor_ctr_x) / anchor_w
        dy = (gt_ctr_y - anchor_ctr_y) / anchor_h
        dw = torch.log(gt_w / anchor_w)
        dh = torch.log(gt_h / anchor_h)

        return torch.stack((dx, dy, dw, dh), dim=1)

    def assign_targets(self, anchors, gt_boxes, iou_pos=0.6, iou_neg=0.3):
        ious = box_iou(anchors, gt_boxes)
        iou_vals, matched_idx = ious.max(dim=1)

        labels = torch.full((anchors.size(0),), -1, dtype=torch.int64, device=anchors.device)
        labels[iou_vals < iou_neg] = 0
        labels[iou_vals >= iou_pos] = 1

        target_boxes = torch.zeros_like(anchors)
        pos_idx = labels == 1
        target_boxes[pos_idx] = self.box_coder_encode(anchors[pos_idx], gt_boxes[matched_idx[pos_idx]])
        return labels, target_boxes

    def forward(self, image, feature_map, targets=None):
        rpn_feat = nn.ReLU()(self.conv(feature_map))
        cls_scores = self.cls_layer(rpn_feat)
        box_preds = self.box_reg_layer(rpn_feat)

        anchors = self.generate_anchors(image, feature_map)
        cls_logits = cls_scores.permute(0, 2, 3, 1).reshape(-1, 1)
        box_preds = box_preds.permute(0, 2, 3, 1).reshape(-1, 4)

        proposals = apply_deltas(box_preds.detach().unsqueeze(1), anchors).squeeze(1)
        proposals, scores = self.filter_proposals(proposals, cls_logits.detach(), image.shape)

        if self.training and targets is not None:
            rpn_labels, rpn_targets = self.assign_targets(anchors, targets[0])
            roi_labels, roi_targets = self.assign_targets(proposals, targets[0], iou_pos=0.5, iou_neg=0.5)
            return rpn_labels, rpn_targets, box_preds, cls_logits, proposals, roi_labels, roi_targets
        else:
            return proposals, scores

# -----------------------
# ROI Head
# -----------------------
class ROIHead(nn.Module):
    def __init__(self, num_classes=2):
        super(ROIHead, self).__init__()
        self.roi_pool = RoIPool((7, 7), spatial_scale=1 / 16)
        self.fc6 = nn.Sequential(nn.Linear(7 * 7 * 512, 4096), nn.ReLU(), nn.Dropout())
        self.fc7 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout())
        self.classifier = nn.Linear(4096, num_classes)
        self.regressor = nn.Linear(4096, 4)

    def forward(self, feature_map, rois):
        pooled = self.roi_pool(feature_map, rois)
        flat = torch.flatten(pooled, start_dim=1)
        x = self.fc6(flat)
        x = self.fc7(x)
        return self.classifier(x), self.regressor(x)

# -----------------------
# Full Faster-RCNN Model
# -----------------------
class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.backbone = Backbone()
        self.rpn = RPN()
        self.roi_head = ROIHead()

    def forward(self, image, targets=None):
        feature_map = self.backbone(image)
        if self.training and targets is not None:
            rpn_labels, rpn_targets, box_preds, cls_logits, proposals, roi_labels, roi_targets = self.rpn(image, feature_map, targets)
            batch_idx = torch.zeros(proposals.size(0), 1, device=proposals.device)
            rois = torch.cat([batch_idx, proposals], dim=1)
            class_logits, bbox_preds = self.roi_head(feature_map, rois)
            return rpn_labels, rpn_targets, box_preds, cls_logits, class_logits, bbox_preds, roi_labels, roi_targets
        else:
            proposals, _ = self.rpn(image, feature_map)
            batch_idx = torch.zeros(proposals.size(0), 1, device=proposals.device)
            rois = torch.cat([batch_idx, proposals], dim=1)
            class_logits, bbox_preds = self.roi_head(feature_map, rois)
            return proposals, class_logits, bbox_preds
    def predict(self, images,target):
        """
        Inference mode — returns probabilities and raw bbox deltas
        """
        self.eval()
        with torch.no_grad():
            proposals, class_logits, bbox_preds = self.forward(images,target)
            probs = torch.softmax(class_logits, dim=1)
        return probs, bbox_preds,proposals
    