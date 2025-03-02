import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import umap
import wandb
from datetime import datetime
import torch
import os
from tqdm import tqdm

from cycler import cycler
from ds import get_celeba 
from Model import get_model
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer

now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

def trainer(args):

    if torch.cuda.is_available():
        device = torch.device("cuda", index=args.idx)
    else:
        device = torch.device("cpu")

    train_ld, valid_ld,test_ld, args = get_celeba(args)

    print(f"#TRAIN Batch: {len(train_ld)}")
    print(f"#VALID Batch: {len(valid_ld)}")
    print(f"#TEST Batch: {len(test_ld)}")
    # if args.log:
    #     run = wandb.init(
    #         project='metriclearning',
    #         config=args,
    #         name=now,
    #         force=True
    #     )

    run_dir = os.getcwd() + '/runs'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    sv_dir = run_dir + f"/{now}"
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)
    best_model_path = sv_dir + f'/best.pt'
    last_model_path = sv_dir + f'/last.pt'
    model = get_model(args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}")
    total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_train_params}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, len(train_ld) * args.epoch)

    old_valid_loss = 1e26
    distance = CosineSimilarity()
    reducer = ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
    #accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    for epoch in range(args.epoch):
        log_dict = {}
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_ld, desc=f"Epoch {epoch+1}/{args.epoch}, Train")
        valid_pbar = tqdm(valid_ld, desc=f"Epoch {epoch+1}/{args.epoch}, Valid")
        
        for _,(img, labels) in enumerate(train_pbar):
            img = img.to(device)
            labels = labels.to(device)
            embeddings = model(img)
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        train_mean_loss = total_loss / len(train_ld)
        
        log_dict['train/loss'] = train_mean_loss

        print(f"Epoch: {epoch} - Train Loss: {train_mean_loss}")
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for _,(img, labels) in enumerate(valid_pbar):
                img = img.to(device)
                labels = labels.to(device)
                embeddings = model(img)
                indices_tuple = mining_func(embeddings, labels)
                loss = loss_func(embeddings, labels, indices_tuple)
                total_loss += loss.item()
                valid_pbar.set_postfix({'loss': loss.item()})
        valid_mean_loss = total_loss / len(valid_ld)
        log_dict['valid/loss'] = valid_mean_loss
        print(f"Epoch: {epoch} - Valid Loss: {valid_mean_loss}")
               
        save_dict = {
            'args' : args,
            'model_state_dict': model.state_dict()
        }
        if valid_mean_loss < old_valid_loss:
            old_valid_loss = valid_mean_loss
            
            torch.save(save_dict, best_model_path)
        torch.save(save_dict, last_model_path)
    #TEST
    check_point = torch.load(sv_dir + f"/best.pt")
    model.load_state_dict(check_point['model_state_dict'])
    logs = []
    model.eval()
    with torch.no_grad():
        batch_cnt = 0
        test_total_loss = 0
        test_correct = 0
        for batch, (test_img, test_label) in enumerate(test_ld):
            batch_cnt = batch
            test_img = test_img.to(device)
            test_label = test_label.to(device)
            pred = model(test_img)

        test_total_loss /= batch_cnt
        test_correct /= len(test_ld.dataset)
         
        print(f"test loss: {test_total_loss} - test acc: {100*test_correct}")
    log_path = sv_dir + "/log.txt"
    with open(log_path, mode='w') as file:
        file.writelines(logs)