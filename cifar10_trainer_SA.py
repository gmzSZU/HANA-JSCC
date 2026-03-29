import os
import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms 
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import random
import network_SA
import modules
import warnings 
import datetime
import math
# =============================================================================
# Randon seed setup
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
# =============================================================================
# BASIC CONFIGURATION
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_batch_size = 256
val_batch_size = 100
num_antenna_tx, num_antenna_rx = 16, 16
save_path = "./results/CIFAR10_MIMO_" + str(num_antenna_tx) + "x" + \
            str(num_antenna_rx) + "_SwinJSCC_wSA_P.pth"

print("Save Path: " + save_path)

best_psnr = 0 #<- THIS will be reset once a training-stage is done
best_epoch = 0
# =============================================================================
transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.ToTensor()
    ])

trainset = torchvision.datasets.CIFAR10(root='./datasets/', train=True, download=False, 
                                        transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=8)
valset = torchvision.datasets.CIFAR10(root='./datasets/', train=False, download=False, 
                                      transform=transform_val)
val_loader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size, shuffle=False, num_workers=8)
# =============================================================================
print("Preparing SemComm Model ...")
model = network_SA.MIMO_JSCC(num_antenna_tx, num_antenna_rx).to(device)
# =============================================================================
print("Preparing Training Configuration ...")
total_epoch = 1000
val_start = total_epoch // 2
lr = 1e-4
min_lr = 1e-6
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.zero_grad()
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=min_lr)
num_iters_train = len(train_loader)
num_iters_val = len(val_loader)
train_snr_list = [i for i in range(1, 10, 2)]
val_snr = 7
val_std = model.SNR2std(val_snr)
snr_est_list = [i for i in range(10, 20, 2)]
# =============================================================================
print("STARTING ...")
start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"./log/training_log_{start_time}.txt"
with open(filename, "w") as f:
    f.write("Number of Tx#Rx Antennas: %d#%d" % (num_antenna_tx, num_antenna_rx))
    f.write("Epoch\tLearning Rate\tPSNR\n")
    for epoch in range(total_epoch):
        model.train()
        total_loss_epoch = 0
        total_psnr_epoch = 0
        print ("\n")
        print ("Epoch: %d || %d, BEST Epoch: %d" % (epoch, total_epoch, best_epoch))
        print ("Current Learning Rate: " + str(optimizer.state_dict()['param_groups'][0]['lr']))
        for imgs, _ in tqdm(train_loader):
            imgs = imgs.to(device)
            train_snr = random.choice(train_snr_list)
            std_est = model.SNR2std(random.choice(snr_est_list))            
            optimizer.zero_grad()

            real = torch.normal(0, math.sqrt(0.5), (imgs.shape[0], num_antenna_rx, num_antenna_tx))
            imag = torch.normal(0, math.sqrt(0.5), (imgs.shape[0], num_antenna_rx, num_antenna_tx))
            train_H = real+1j*imag
            train_H = train_H.to(device)

            res_recovery = model(imgs, train_snr, train_H)
            loss = loss_fn(res_recovery, imgs)
            loss.backward()
            optimizer.step()
            
            psnr_avg = modules.Compute_batch_PSNR(imgs.cpu().detach().numpy(), res_recovery.cpu().detach().numpy())
            total_psnr_epoch += psnr_avg
            total_loss_epoch += loss.item()
            
        print("Epoch: %d || Avg Train Loss: %.05f || AVG PSNR: %.05f" % (epoch, total_loss_epoch/num_iters_train, total_psnr_epoch/num_iters_train)) 
        f.write("Epoch: %d || Learning Rate: %.07f || Avg Train Loss: %.05f || AVG Train PSNR: %.05f || " % \
                (epoch, optimizer.state_dict()['param_groups'][0]['lr'], total_loss_epoch/num_iters_train, total_psnr_epoch/num_iters_train))
        scheduler.step()
        
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            val_loss = 0
            for imgs, _ in tqdm(val_loader):
                imgs = imgs.to(device)
                std_est = model.SNR2std(20)            
  
                real = torch.normal(0, math.sqrt(0.5), (val_batch_size, num_antenna_rx, num_antenna_tx))
                imag = torch.normal(0, math.sqrt(0.5), (val_batch_size, num_antenna_rx, num_antenna_tx))
                val_H = real+1j*imag
                val_H = val_H.to(device)  
  
                res_recovery = model(imgs, val_snr, val_H)
                loss = loss_fn(res_recovery, imgs)                
                val_loss += loss.item()
                psnr_avg = modules.Compute_batch_PSNR(imgs.cpu().detach().numpy(), res_recovery.cpu().detach().numpy())
                total_psnr += psnr_avg
            print("Val Result: Avg Val Loss: %.05f || Avg PSNR: %.05f" % (val_loss/num_iters_val, total_psnr/num_iters_val))
            f.write("AVG Val PSNR: %.05f \n" % (total_psnr/num_iters_val))
            
            if (total_psnr/num_iters_val) > best_psnr:
                best_psnr = total_psnr/num_iters_val
                best_epoch = epoch
                print ("New Record Confirm, Saving Model...")
                torch.save(model.state_dict(), save_path)

    f.write("Training process for CLASSIC MIMO SemComm System is OVER.\n")
    f.write("Sub-optimal PSNR: %.5f \n" % best_psnr)
    f.write("Corresponding Epoch: %d \n" % best_epoch)
    f.write("Save Path: " + save_path)
    f.write("\n")

print ("Training process for CLASSIC-MIMO SemComm System is OVER.")
print ("Sub-optimal PSNR: %.5f" % best_psnr)
print ("Corresponding Epoch: %d" % best_epoch)
print ("Save Path: " + save_path)
print ("\n")
        



