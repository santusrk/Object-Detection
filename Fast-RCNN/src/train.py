import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import FastRCNN
from dataset import FastRCNNDataset
import torch.nn.functional as F


# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = os.getcwd()
proposals_path = os.path.join(root_dir,'Fast-RCNN/data/region_proposals.pkl')
image_folder = os.path.join(root_dir,'Fast-RCNN/resized_images')




def fast_rcnn_collate_fn(batch):
    """
    Pads all images in a batch to the same size and stacks them.
    Returns:
        padded_images: Tensor [B, 3, max_H, max_W]
        proposals: list of proposals per image
    """
    images, all_proposals = zip(*batch)  # unzip batch
    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]

    max_height = max(heights)
    max_width = max(widths)

    padded_images = []
    for img in images:
        # img shape: [3, H, W]
        pad_bottom = max_height - img.shape[1]
        pad_right = max_width - img.shape[2]
        padded = F.pad(img, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
        padded_images.append(padded)

    padded_images = torch.stack(padded_images)  # now all [3, max_H, max_W]

    return padded_images, all_proposals



# Hyperparameters
BATCH_SIZE = 2 
NUM_EPOCHS = 70
LEARNING_RATE = 0.01
MOMENTUM = 0.9


# Transforms
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = FastRCNNDataset(proposals_path, image_folder, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fast_rcnn_collate_fn)



# Model
model = FastRCNN(num_classes=2).to(DEVICE)

# Loss and Optimizer
cls_criterion = nn.CrossEntropyLoss()
reg_criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# LR Shedulingoptimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=3)

best_loss = float('inf')
early_stop_counter = 0
patience = 7

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    for image_batch, proposals_batch in train_loader:
        images = []
        rois = []
        labels = []
        target_bboxes = []

        for batch_idx, (image_tensor, proposals) in enumerate(zip(image_batch, proposals_batch)):
            image_tensor = image_tensor.to(DEVICE)
            images.append(image_tensor)

            for p in proposals:
                rois.append([batch_idx] + p['region_proposal_box'])
                labels.append(p['label'])
                target_bboxes.append(p['target_box'])

        images = torch.stack(images).to(DEVICE)
        rois_tensor = torch.tensor(rois, dtype=torch.float32).to(DEVICE)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(DEVICE)
        target_tensor = torch.tensor(target_bboxes, dtype=torch.float32).to(DEVICE)

        # Forward + Loss
        class_logits, pred_bbox = model(images, rois_tensor)
        cls_loss = cls_criterion(class_logits, labels_tensor)

        pos_idx = [i for i, l in enumerate(labels) if l == 1]

        if pos_idx:
            pred_positive = pred_bbox[pos_idx]
            target_positive = target_tensor[pos_idx]
            reg_loss = reg_criterion(pred_positive, target_positive)
        else:
            reg_loss = torch.tensor(0.0).to(DEVICE)
                

        # Backpropagation
        loss = cls_loss + reg_loss
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()

    epoch_loss = total_cls_loss + total_reg_loss
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - "
          f"Classification Loss: {total_cls_loss:.4f}, "
          f"Regression Loss: {total_reg_loss:.4f}")

    # Step LR scheduler
    scheduler.step(epoch_loss)

    # Early stopping check
    if epoch_loss < best_loss - 1e-4:
        best_loss = epoch_loss
        early_stop_counter = 0

        # Save best model
        save_path = os.path.join("saved_models", "fast_rcnn_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Model improved. Saved to {save_path}")
    else:
        early_stop_counter += 1
        print(f"[INFO] No improvement. Early stop patience: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience: 
        print("[INFO] Early stopping triggered.")
        break

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Classification Loss: {total_cls_loss:.4f}, Regression Loss: {total_reg_loss:.4f}")

