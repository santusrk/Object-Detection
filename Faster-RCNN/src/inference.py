import torch
import torchvision.transforms as T
import cv2
import os
from model import FasterRCNN  # make sure this matches your structure
from utils import apply_deltas,NMS
from torchvision.ops import nms

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model ---
model = FasterRCNN().to(device)
model_path = 'C:/Users/Lenovo/Downloads/Computer_Vision/Object-Detection/saved_models/faster_rcnn_best.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Preprocessing ---
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def infer(image_path, target):
    # Load and preprocess image
    orig = cv2.imread(image_path)
    if orig is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image).unsqueeze(0).to(device)  # shape [1, 3, H, W]

    with torch.no_grad():
        # Run full model forward pass
        probs, bbox_preds,proposals= model.predict(input_tensor, [torch.tensor(target).to(device)])
        scores = torch.softmax(probs, dim=1)[:, 1]  # foreground class scores
        pred_boxes_tensor = torch.tensor(bbox_preds, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)        
        keep = scores > 0.4
        if keep.sum() == 0:
            print("[INFO] No proposals passed the threshold.")
            return
        box_pred = pred_boxes_tensor[keep]
        target_box = proposals[keep]
        final_boxes = apply_deltas(box_pred, target_box)     
        final_boxes = final_boxes.cpu().numpy().astype(int) 
        predicted_box,max_ious = NMS(torch.tensor(target),final_boxes,len(target))
        print(predicted_box)

    for i, box in enumerate(predicted_box):
        x1, y1, x2, y2 = map(int, box)
        p1,q1,p2,q2 = map(int,target[i])
        score = max_ious[i] 
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 255), 2) # predicted
        cv2.rectangle(orig, (p1, q1), (p2, q2), (255, 140, 255), 2) # original
        cv2.putText(orig, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)




    cv2.imshow("Predictions", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import pickle
# --- Run on Sample Image ---
if __name__ == "__main__":
    test_image="FudanPed00007.png"
    image_folder = "C:/Users/Lenovo/Downloads/Computer_Vision/Object-Detection/Faster-RCNN/data/resized_images"  
    test_path = os.path.join(image_folder,test_image)
    gt_path = "C:/Users/Lenovo/Downloads/Computer_Vision/Object-Detection/Faster-RCNN/data/scaled_gt_box.pkl"
    with open(gt_path, 'rb') as f:
        gt_data = pickle.load(f)
    gt_box = [p for p in gt_data if p['filename']==test_image][0]['ground_truth_box']
    print(gt_box)
    infer(test_path,gt_box)
   
