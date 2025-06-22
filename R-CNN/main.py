import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.data_prerpcessing import process_image
from src.inference import inference_with_nms
from src.visualize import visualize
import cv2
import joblib
from tensorflow.keras.models import load_model
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))



ROOT_DIR= os.getcwd()
image_folder = os.path.join(ROOT_DIR,'PennFudanPed/PNGImages')
annot_folder = os.path.join(ROOT_DIR,'PennFudanPed/boundary_box')
model_path =os.path.join(ROOT_DIR,'R-CNN/src')
fc2_model_path = os.path.join(model_path,'fc2_model.h5')
clf_path = os.path.join(model_path,'svm_classifier.pkl')
reg_path = os.path.join(model_path,'svr_regressor.pkl')
fc2_model= load_model(fc2_model_path, compile=False)
clf = joblib.load(clf_path)
reg = joblib.load(reg_path)

if __name__=="__main__":
    filename ="FudanPed00008.png"
    image = cv2.imread(os.path.join(image_folder,filename))
    train_image,train_label,region_proposal,gt_bbox,target_bbox= process_image(filename)
    final_boxes, final_scores = inference_with_nms(image, region_proposal, clf, reg, fc2_model)
    visualize(image, final_boxes, gt_bbox, label='object', scores=final_scores)


