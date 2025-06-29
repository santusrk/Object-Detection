import pandas as pd
import numpy as np
import os
import cv2
import json
import pickle
from tqdm import tqdm

root_dir = os.getcwd()
image_folder = os.path.join(root_dir, 'PennFudanPed/PNGImages')
bbox_folder = os.path.join(root_dir, 'PennFudanPed/boundary_box')
output_dir = os.path.join(root_dir, 'Faster-RCNN/data')
os.makedirs(output_dir,exist_ok=True)
resized_image_folder = os.path.join(output_dir, 'resized_images')
os.makedirs(resized_image_folder, exist_ok=True)

def resize_image(image, short_side=600):
    h, w = image.shape[:2]
    scale = short_side / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_img, scale

def process_image(filename):

    try:
        image_path = os.path.join(image_folder, filename)
        im = cv2.imread(image_path)
        if im is None or im.shape[0] == 0 or im.shape[1] == 0:
            print(f"Skipping corrupt or empty image: {filename}")
            return []

        annot_file = os.path.join(bbox_folder, filename.replace('.png', '.json'))
        with open(annot_file, 'r') as f:
            annot = json.load(f)

        ground_truth_bboxes = annot['boundary_box']
        resized_im, scale = resize_image(im, short_side=600)

        # Save resized image to disk
        resized_path = os.path.join(resized_image_folder, filename)
        cv2.imwrite(resized_path, resized_im)

        # Scale ground truth boxes
        scaled_gt_boxes = [[x1 * scale, y1 * scale, x2 * scale, y2 * scale] for (x1, y1, x2, y2) in ground_truth_bboxes]
        annotation ={}
        annotation['filename']=filename
        annotation['ground_truth_box']=scaled_gt_boxes
    except Exception as e:
        print(f"Error in {filename}: {e}")

    return annotation

def data_process():
    scaled_boxes = []

    for filename in tqdm(os.listdir(image_folder)):
        if filename.endswith('.png'):
            annotation = process_image(filename)
            scaled_boxes.append(annotation)

    # Save region proposals
    with open(os.path.join(output_dir, 'scaled_gt_box.pkl'), 'wb') as f:
        pickle.dump(scaled_boxes, f)

    print(f"Total region proposals: {len(scaled_boxes)}")
    print(f"Saved region proposals to: {os.path.join(output_dir, 'scaled_gt_box.pkl')}")
    print(f"Resized images saved to: {resized_image_folder}")

if __name__ == "__main__":
    data_process()
