import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import FasterRCNN
from dataset import FasterRCNNDataset  # Make sure this is implemented properly
import os
from torchvision import transforms as T
import torch.optim as optim
import pickle

# ---------------------------
# Hyperparameters & Config
# ---------------------------
BATCH_SIZE = 1
NUM_EPOCHS = 70
LEARNING_RATE = 0.01
MOMENTUM = 0.9
PATIENCE = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Dataset & Dataloader
# ---------------------------
image_folder = "./Faster-RCNN/data/resized_images"
ground_truth_path = "./Faster-RCNN/data/scaled_gt_box.pkl"

transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

dataset = FasterRCNNDataset(image_folder, ground_truth_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------
# Model, Loss, Optimizer
# ---------------------------
model = FasterRCNN().to(device)
criterion_rpn_cls = nn.BCEWithLogitsLoss()
cls_criterion = nn.CrossEntropyLoss()
reg_criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# ---------------------------
# Training Loop
# ---------------------------
best_loss = float('inf')
early_stop_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_rpn_loss = 0.0
    total_roi_loss = 0.0

    for images, targets in data_loader:
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        # Forward pass
        outputs = model(images, targets)
        (rpn_labels, rpn_targets, box_preds, cls_logits,
         roi_class_logits, roi_bbox_preds, roi_labels, roi_targets) = outputs

        # ---------------------
        # RPN Loss
        # ---------------------
        rpn_labels = rpn_labels.float()  # Ensure target is long
        cls_logits = cls_logits.float() 
        rpn_labels = rpn_labels.squeeze()
        cls_logits = cls_logits.squeeze()
        

        valid_rpn = rpn_labels != -1
        cls_loss_rpn = criterion_rpn_cls(cls_logits[valid_rpn], rpn_labels[valid_rpn])
        pos_rpn = rpn_labels == 1
        reg_loss_rpn = reg_criterion(box_preds[pos_rpn], rpn_targets[pos_rpn]) if pos_rpn.sum() > 0 else torch.tensor(0.0, device=device)

        rpn_loss = cls_loss_rpn + reg_loss_rpn

        # ---------------------
        # ROI Head Loss
        # ---------------------
        roi_labels = roi_labels.squeeze()
        roi_class_logits = roi_class_logits.squeeze()

        cls_loss_roi = cls_criterion(roi_class_logits, roi_labels)

        pos_roi = roi_labels == 1
        reg_loss_roi = reg_criterion(roi_bbox_preds[pos_roi], roi_targets[pos_roi]) if pos_roi.sum() > 0 else torch.tensor(0.0, device=device)

        roi_loss = cls_loss_roi + reg_loss_roi

        # ---------------------
        # Total Loss & Backprop
        # ---------------------
        total_loss = rpn_loss + roi_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_rpn_loss += rpn_loss.item()
        total_roi_loss += roi_loss.item()

    # Epoch Summary
    epoch_loss = total_rpn_loss + total_roi_loss
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] - RPN Loss: {total_rpn_loss:.4f} - ROI Loss: {total_roi_loss:.4f} - Total: {epoch_loss:.4f}")

    # Learning rate scheduling
    scheduler.step(epoch_loss)

    # Save best model
    if epoch_loss < best_loss - 1e-4:
        best_loss = epoch_loss
        early_stop_counter = 0
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), "saved_models/faster_rcnn_best.pth")
        print("[✓] Model improved. Saved.")
    else:
        early_stop_counter += 1
        print(f"[✗] No improvement. Early Stop Counter: {early_stop_counter}/{PATIENCE}")

    if early_stop_counter >= PATIENCE:
        print("[⚠️] Early stopping triggered.")
        break
