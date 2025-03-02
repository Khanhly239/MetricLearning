import os
import torch
from torch import nn
from Model import get_model
from torch.utils.data import DataLoader
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', action="store_true")
    parser.add_argument('--custom', action="store_true")
    parser.add_argument('--wk', default=64, type = int)
    parser.add_argument('--task', default='acb-inv-cls',
                        choices=[
                            "acb-inv-cls",
                            "gttt-cls",
                            "gotit-inv-cls"
                        ])
    args = parser.parse_args()
    date = "07_25_2024_10_33_46"
    model_abs_path = os.getcwd() + f"/run/{args.task}/{date}/best.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(3, 2, weight = args.weight, custom=args.custom).to(device=device)
    
    checkpoint = torch.load(model_abs_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    from ds.celebA import get_celeba
    train_data, valid_data, test_data = get_celeba(size = (args.h, args.w))
    test_ld = DataLoader(
        test_data, 
        batch_size=1, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=args.wk
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    logs = []
    model.eval()
    with torch.no_grad():
        batch_cnt = 0
        test_total_loss = 0
        test_correct = 0
        for batch, (test_img, test_label, test_path) in enumerate(test_ld):
            batch_cnt = batch
            test_img = test_img.to(device)
            test_label = test_label.to(device)
            
            pred = model(test_img)
            loss = loss_fn(pred, test_label)
            
            test_total_loss += loss.item()
            test_correct += (pred.argmax(1) == test_label).type(torch.float).sum().item()
                    
            short_path = "/".join(test_path[0].split("/")[-4:])
            
            logs.append(
                f"Pred: {pred.argmax(1).item()} - Truth: {test_label.item()} - Path: {short_path}\n"
            )
    
        test_total_loss /= batch_cnt
        test_correct /= len(test_ld.dataset)
        
        print(f"test loss: {test_total_loss} - test acc: {100*test_correct}")
    
    log_path = "/".join(model_abs_path.split("/")[:-1]) + "/log.txt"
    with open(log_path, mode='w') as file:
        file.writelines(
            logs
        )