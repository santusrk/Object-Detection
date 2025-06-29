from torch.utils.data import Dataset
import os
import cv2
import pickle
import torch

class FasterRCNNDataset(Dataset):
    def __init__(self, image_folder, gt_path, transform=None):
        with open(gt_path, 'rb') as f:
            self.gt_data = pickle.load(f)
        
        self.image_folder = image_folder
        self.transform = transform
        self.filename = sorted(os.listdir(self.image_folder))  # Sorted for consistency

    def __len__(self):
        return len(self.filename)
    
    def __getitem__(self, index): 
        img_name = self.filename[index]
        img_path = os.path.join(self.image_folder, img_name)

        # Get ground truth box
        gt = next(item for item in self.gt_data if item['filename'] == img_name)
        bounding_box = torch.tensor(gt['ground_truth_box'], dtype=torch.float32)  # Ensure torch.Tensor

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, bounding_box

    

