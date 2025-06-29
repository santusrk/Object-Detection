import torch
import numpy as np
from torchvision.ops import box_iou





def get_iou(bb1,bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1 ['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2 ['y2']
    x_left = max(bb1['x1'],bb2['x1'])
    x_right = min(bb1['x2'],bb2['x2'])
    y_top = min(bb1['y2'],bb2['y2'])
    y_bottom = max(bb1['y1'],bb2['y1'])
    if x_right < x_left or y_top<y_bottom :
        return 0.0
    intersection_area = (x_right-x_left)*(y_top-y_bottom)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area/float(bb1_area+bb2_area-intersection_area)
    iou = max(0.0, min(1.0, iou))  # keep in [0, 1]
    assert iou >=0.0
    assert iou <=1.0
    return iou




def get_iou1(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def get_target_bbox(gt, prop):
    px, py, pw, ph = prop_to_cxcywh(prop)
    gx, gy, gw, gh = prop_to_cxcywh(gt)

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = np.log(gw / pw)
    th = np.log(gh / ph)
    return [tx, ty, tw, th]

def prop_to_cxcywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h

def apply_deltas(box_transform_pred,anchors):
    box_transform_pred=box_transform_pred.reshape(box_transform_pred.shape[0],-1,4)
    w=anchors[:,2]-anchors[:,0]
    h= anchors[:,3] - anchors[:,1]
    center_x = anchors[:,0] + 0.5*w
    center_y = anchors[:,1] + 0.5*h

    dx= box_transform_pred[...,0]
    dy= box_transform_pred[...,1]
    dw= box_transform_pred[...,2]
    dh= box_transform_pred[...,3]

    pred_center_x = dx*w[:,None] + center_x[:,None]
    pred_center_y = dy*h[:,None] + center_y[:,None]
    pred_w=torch.exp(dw)*w[:,None]
    pred_h=torch.exp(dh)*h[:,None]

    pred_box_x1= pred_center_x - 0.5*pred_w
    pred_box_y1= pred_center_y -  0.5*pred_h
    pred_box_x2= pred_center_x + 0.5*pred_w
    pred_box_y2= pred_center_y + 0.5*pred_h
    pred_boxes = torch.stack((pred_box_x1,pred_box_y1,pred_box_x2,pred_box_y2),dim=2)
    return pred_boxes




def apply_deltas2(prop, deltas):
    assert len(prop) == 4
    assert len(deltas) == 4 
    x1, y1, x2, y2 = prop
    prop_w = x2 - x1
    prop_h = y2 - y1
    prop_ctr_x = x1 + 0.5 * prop_w
    prop_ctr_y = y1 + 0.5 * prop_h

    tx, ty, tw, th = deltas
    pred_ctr_x = prop_ctr_x + tx * prop_w
    pred_ctr_y = prop_ctr_y + ty * prop_h
    pred_w = np.exp(tw) * prop_w
    pred_h = np.exp(th) * prop_h

    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    return [int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)]



def NMS(gt_boxes, pred_boxes, top_k=2):
    # Convert to tensor and float32
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
    else:
        gt_boxes = gt_boxes.to(dtype=torch.float32)

    if isinstance(pred_boxes, np.ndarray):
        pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
    else:
        pred_boxes = pred_boxes.to(dtype=torch.float32)

    # Fix shape if needed: [N, 1, 4] → [N, 4]
    if gt_boxes.ndim == 3:
        gt_boxes = gt_boxes.squeeze(1)
    if pred_boxes.ndim == 3:
        pred_boxes = pred_boxes.squeeze(1)

    # Ensure correct shape
    assert gt_boxes.shape[1] == 4, f"gt_boxes shape is {gt_boxes.shape}, expected [N, 4]"
    assert pred_boxes.shape[1] == 4, f"pred_boxes shape is {pred_boxes.shape}, expected [N, 4]"

    # Compute IoU
    ious = box_iou(pred_boxes, gt_boxes) 
    top_idx=[]
    max_ious=[]
    for i in range(ious.shape[1]):
        max_iou =max(ious[:,i])    
        idx = torch.argmax(ious[:,i]).item()
        top_idx.append(idx)
        max_ious.append(max_iou)
    return pred_boxes[top_idx], max_ious





           
