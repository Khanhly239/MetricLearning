import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from rich.progress import track
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity as cossim
import pickle
from glob import glob
import albumentations as A
import json
from shutil import copy
import argparse

from ds.celebA import get_celeba
from ds import super_transform

import torch
from torch import nn
from Model import get_model
from torch.utils.data import DataLoader
from torchvision import transforms

def test_roc(embedding_list, label_list):
    test_embs = np.array(embedding_list)
    test_label = np.array(label_list)

    scores = cossim(test_embs, test_embs)
    label_mat = test_label.reshape([-1,1]) == test_label.reshape([1,-1])
    
    impostor = np.sum(1 - label_mat)
    genuine = np.sum(label_mat)
    
    negative_scores = scores[label_mat == False]
    positive_scores = scores[label_mat-np.identity(len(embedding_list)) == True]
    
    def score(threshold):
        accept = (scores > threshold).astype(np.float32)                
        FA = np.sum((label_mat - accept)<0)
        FR = np.sum((label_mat - accept)>0)
        return FA/impostor, FR/(genuine-len(embedding_list))
    
    high = 1
    low = -1
    EER, threshold = None, None

    while abs(high - low) > 1e-7:
        middle = (high + low)/2
        far, frr = score(middle)
        print(middle, far, frr)
        threshold = middle
        EER = far
        if frr > far:
            high = middle
        else:
            low = middle
    return EER, threshold, np.mean(negative_scores), np.mean(positive_scores)

def resize(image):
    size = (512, 256)
    h, w = image.shape[:2]
    w_new = size[1]
    h_new = round(h / w * w_new)
    resized_image = cv2.resize(image, (w_new, h_new))
    h_limit = size[0]
    
    if resized_image.shape[0] > h_limit:
        upper = resized_image[:h_limit // 2, :]
        lower = resized_image[-h_limit // 2:, :]
        resized_image = np.concatenate([upper, lower], axis=0)
        
    elif resized_image.shape[0] < h_limit:
        resized_image = cv2.copyMakeBorder(resized_image,0,h_limit - resized_image.shape[0],
                                               0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
    return resized_image

out_transform = [
    A.Resize(512, 256),
    A.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    ),
]

target_transform = A.Compose(
    out_transform
)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--k', type = int, default = 1)
    parser.add_argument('--th', type = float, default = 0.8)
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--max', action='store_true')
    parser.add_argument('--rank', action='store_true')
    parser.add_argument('--method', type = str, default='eer', choices = ['eer', 'nearest'])
    parser.add_argument('--alpha', type=float, default=0.9)
    
    args = parser.parse_args()
    
    train_data, valid_data, test_data, _, args = get_celeba()
    train_data.img_transform = super_transform(train = False)
    num_class = len(args)
    print(f"Number of class: {num_class}")

    args = {
        args[key]: key for key in args
    }
    
    embedding_size = 128

    date = "26_07_2024" 
    model_abs_path = os.getcwd() + f"/run/{args.task}/{date}/best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = 0)

    model = get_model(3, num_class, weight = True, custom = -1, embedding = embedding_size).to(device=device)

    checkpoint = torch.load(model_abs_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    