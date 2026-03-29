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
import network
import modules
import warnings 
import datetime
from distil_util import nkd_latent_loss
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

teacher_path = "./results/CIFAR10_MIMO_" + str(num_antenna_tx) + "x" + \
                str(num_antenna_rx) + "_HANA_JSCC.pth"

student_path = "./results/CIFAR10_MIMO_" + str(num_antenna_tx) + "x" + \
                str(num_antenna_rx) + "_HANA_JSCC.pth"

                
save_path = "./results/CIFAR10_MIMO_" + str(num_antenna_tx) + "x" + \
                str(num_antenna_rx) + "_HANA_JSCC_distilled.pth"

print("Save Path: " + save_path)
best_psnr = 0 #<- THIS will be reset once a training-stage is done
best_epoch = 0
# =============================================================================
transform_train = transforms.Compose([
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
print("Preparing Teacher Model ...")
teacher_model = network.MIMO_JSCC(num_antenna_tx, num_antenna_rx).to(device)
msg = teacher_model.load_state_dict(torch.load(teacher_path, map_location='cpu'), strict=False)
print (msg)
teacher_model.to(device)
teacher_model.eval()
print("Freezing Teacher ...")
for param in teacher_model.parameters():
    param.requires_grad = False
# =============================================================================
print("Preparing Student Model ...")
student_model = network.MIMO_JSCC(num_antenna_tx, num_antenna_rx).to(device)
msg = student_model.load_state_dict(torch.load(student_path, map_location='cpu'), strict=False)
print (msg)
student_model.to(device)

print("Freezing Semantic Encoder ...")
for param in student_model.semantic_encoder.parameters():
    param.requires_grad = False
print("Freezing SNR-Adaptation Module (Tx) ...")
for param in student_model.bm_tx_list.parameters():
    param.requires_grad = False
for param in student_model.sm_tx_list.parameters():
    param.requires_grad = False

print("Freezing Channel Encoder ...")
for param in student_model.channel_encoder_p1.parameters():
    param.requires_grad = False
for param in student_model.channel_encoder_p2.parameters():
    param.requires_grad = False
print("Freezing Channel Gain Adaptation Module (Tx) ...")
for param in student_model.prec_fc.parameters():
    param.requires_grad = False
for param in student_model.prec_TF_layers.parameters():
    param.requires_grad = False


print("Freezing SNR-Adaptation Module (Rx) ...")
for param in student_model.bm_rx_list.parameters():
    param.requires_grad = False
for param in student_model.sm_rx_list.parameters():
    param.requires_grad = False
print("Freezing Semantic Decoder ...")
for param in student_model.semantic_decoder.parameters():
    param.requires_grad = False

# =============================================================================
print("Preparing Training Configuration ...")
total_epoch = 2000
val_start = total_epoch // 2
lr = 1e-5
min_lr = 1e-6
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
optimizer.zero_grad()
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=min_lr)
num_iters_train = len(train_loader)
num_iters_val = len(val_loader)
train_snr_list = [i for i in range(1, 10, 2)]
val_snr = 7
val_std = student_model.SNR2std(val_snr)
sigma_err_min = 0.01
sigma_err_max = 0.1
beta = 5e-2
print("Beta: %.05f" % (beta))
# =============================================================================
print("STARTING ...")
start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"./log/training_log_{start_time}.txt"
with open(filename, "w") as f:
    f.write("Number of Tx#Rx Antennas: %d#%d" % (num_antenna_tx, num_antenna_rx))
    f.write("Epoch\tLearning Rate\tPSNR\n")
    for epoch in range(total_epoch): 
        student_model.train()
        total_loss_epoch = 0
        total_psnr_epoch = 0
        total_cos_loss_epoch = 0
        print ("\n")
        print ("Epoch: %d || %d, BEST Epoch: %d" % (epoch, total_epoch, best_epoch))
        print ("Current Learning Rate: " + str(optimizer.state_dict()['param_groups'][0]['lr']))
        for imgs, _ in tqdm(train_loader):
            imgs = imgs.to(device)
            train_snr = random.choice(train_snr_list)
            var_est = random.uniform(sigma_err_min, sigma_err_max)   
            optimizer.zero_grad()

            real = torch.normal(0, math.sqrt(0.5), (imgs.shape[0], num_antenna_rx, num_antenna_tx))
            imag = torch.normal(0, math.sqrt(0.5), (imgs.shape[0], num_antenna_rx, num_antenna_tx))
            train_H = real+1j*imag
            train_H = train_H.to(device)

            real = torch.normal(0, math.sqrt(var_est/2), (imgs.shape[0], num_antenna_rx, num_antenna_tx))
            imag = torch.normal(0, math.sqrt(var_est/2), (imgs.shape[0], num_antenna_rx, num_antenna_tx))
            train_H_delta = real+1j*imag
            train_H_delta = train_H_delta.to(device)  
            
            B = imgs.shape[0]
            
            # Student forward
            H_est = train_H + train_H_delta
            # Encoding
            z = student_model.semantic_encoder(imgs)
            # Channel Adaptation to Additive Noise (From SwinJSCC, thanks)
            snr_cuda = torch.tensor(train_snr, dtype=torch.float).to(device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(student_model.SA_layer_num):
                if i == 0:
                    temp = student_model.sm_tx_list[i](z.detach())
                else:
                    temp = student_model.sm_tx_list[i](temp)

                bm = student_model.bm_tx_list[i](snr_batch).unsqueeze(1).expand(-1, 64, -1)
                temp = temp * bm
            mod_val = student_model.sigmoid(student_model.sm_tx_list[-1](temp))
            z = z * mod_val    
            # Feature Compression (Phase-I)
            z = student_model.channel_encoder_p1(z.view(B, student_model.num_antenna_tx, -1))
            # Channel Gain Adaptation (TF Version)  
            H_est_4fc = torch.cat((H_est.real.unsqueeze(1), H_est.imag.unsqueeze(1)), dim=-1)
            H_est_4fc = H_est_4fc.view(B, 1, -1)
            csi_token_prec = student_model.prec_fc(H_est_4fc)
            z = torch.cat((csi_token_prec, z), dim=1)
            z = student_model.prec_TF_layers(z)
            # Projection and Feature Compression (Phase-II)
            z = student_model.channel_encoder_p2(z[:, 1:, :])
            # Power Normalization
            z = z.view(B, student_model.num_antenna_tx, -1, 2)        
            z = z[:, :, :, 0] + 1j*z[:, :, :, 1]   
            z = student_model.Power_norm(z)  
            # Imperfect SVD-based Precoding
            U, _, VH = torch.linalg.svd(H_est, full_matrices=False)
            V = torch.conj(VH).transpose(-1, -2)
            Vs = student_model.Decomposed_Complex(V, z)
            # MIMO Transmission
            y = student_model.Decomposed_Complex(train_H, Vs)       
            # Additive Noise Generation
            std = student_model.SNR2std(train_snr)
            N = torch.normal(0, std/2, y.shape) + 1j*torch.normal(0, std/2, y.shape)
            N = N.to(device)
            y = y + N
            # Imperfect SVD-based Post-processing
            UH = torch.conj(U).transpose(-1, -2)
            Uy = student_model.Decomposed_Complex(UH, y)   
            Uy = torch.cat((Uy.real, Uy.imag), dim=-1)
            # Feature Reconstruction (Phase-I)
            Uy = student_model.channel_decoder_p1(Uy)
            # Channel Gain Adaptation (TF Version)  
            csi_token_postp = student_model.postp_fc(H_est_4fc)
            Uy = torch.cat((csi_token_postp, Uy), dim=1)
            Uy = student_model.postp_TF_layers(Uy)
            # Projection and Feature Reconstruction (Phase-II)
            Uy_st = student_model.channel_decoder_p2(Uy[:, 1:, :]).view(B, 64, 256)            
            # Channel Adaptation to Additive Noise (From SwinJSCC, thanks)
            snr_cuda = torch.tensor(train_snr, dtype=torch.float).to(device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(student_model.SA_layer_num):
                if i == 0:
                    temp = student_model.sm_rx_list[i](Uy_st.detach())
                else:
                    temp = student_model.sm_rx_list[i](temp)
                bm = student_model.bm_rx_list[i](snr_batch).unsqueeze(1).expand(-1, 64, -1)
                temp = temp * bm
            mod_val = student_model.sigmoid(student_model.sm_rx_list[-1](temp))
            Uy = Uy_st * mod_val
            # Semantic Decoding
            res_recovery = student_model.semantic_decoder(Uy)
            
            # Teacher forward (SwinJSCC-CGASA)
            # Encoding
            z = teacher_model.semantic_encoder(imgs)
            # Channel Adaptation to Additive Noise (From SwinJSCC, thanks)
            snr_cuda = torch.tensor(train_snr, dtype=torch.float).to(device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(teacher_model.SA_layer_num):
                if i == 0:
                    temp = teacher_model.sm_tx_list[i](z.detach())
                else:
                    temp = teacher_model.sm_tx_list[i](temp)

                bm = teacher_model.bm_tx_list[i](snr_batch).unsqueeze(1).expand(-1, 64, -1)
                temp = temp * bm
            mod_val = teacher_model.sigmoid(teacher_model.sm_tx_list[-1](temp))
            z = z * mod_val    
            # Feature Compression (Phase-I)
            z = teacher_model.channel_encoder_p1(z.view(B, teacher_model.num_antenna_tx, -1))
            # Channel Gain Adaptation (TF Version)  
            H_est_4fc = torch.cat((train_H.real.unsqueeze(1), train_H.imag.unsqueeze(1)), dim=-1)
            H_est_4fc = H_est_4fc.view(B, 1, -1)
            csi_token_prec = teacher_model.prec_fc(H_est_4fc)
            z = torch.cat((csi_token_prec, z), dim=1)
            z = teacher_model.prec_TF_layers(z)
            # Projection and Feature Compression (Phase-II)
            z = teacher_model.channel_encoder_p2(z[:, 1:, :])
            # Power Normalization
            z = z.view(B, teacher_model.num_antenna_tx, -1, 2)        
            z = z[:, :, :, 0] + 1j*z[:, :, :, 1]   
            z = teacher_model.Power_norm(z)  
            # Imperfect SVD-based Precoding
            U, _, VH = torch.linalg.svd(train_H, full_matrices=False)
            V = torch.conj(VH).transpose(-1, -2)
            Vs = teacher_model.Decomposed_Complex(V, z)
            # MIMO Transmission
            y = teacher_model.Decomposed_Complex(train_H, Vs)       
            # Additive Noise Generation
            y = y + N
            # Imperfect SVD-based Post-processing
            UH = torch.conj(U).transpose(-1, -2)
            Uy = teacher_model.Decomposed_Complex(UH, y)   
            Uy = torch.cat((Uy.real, Uy.imag), dim=-1)
            # Feature Reconstruction (Phase-I)
            Uy = teacher_model.channel_decoder_p1(Uy)
            # Channel Gain Adaptation (TF Version)  
            csi_token_postp = teacher_model.postp_fc(H_est_4fc)
            Uy = torch.cat((csi_token_postp, Uy), dim=1)
            Uy = teacher_model.postp_TF_layers(Uy)
            # Projection and Feature Reconstruction (Phase-II)
            Uy_tc = teacher_model.channel_decoder_p2(Uy[:, 1:, :]).view(B, 64, 256)        
            
            cos_loss = nkd_latent_loss(Uy_st, Uy_tc)
            
            loss = loss_fn(res_recovery, imgs) + beta*cos_loss
            loss.backward()
            optimizer.step()
            
            psnr_avg = modules.Compute_batch_PSNR(imgs.cpu().detach().numpy(), res_recovery.cpu().detach().numpy())
            total_psnr_epoch += psnr_avg
            total_loss_epoch += loss.item()
            total_cos_loss_epoch += beta*cos_loss
            
        print("Epoch: %d || Avg Train Loss: %.05f || AVG PSNR: %.05f" % (epoch, total_loss_epoch/num_iters_train, total_psnr_epoch/num_iters_train)) 
        f.write("Epoch: %d || Learning Rate: %.07f || Avg Train Loss: %.05f || AVG Train PSNR: %.05f || AVG CosLoss: %.05f" % \
                (epoch, optimizer.state_dict()['param_groups'][0]['lr'], total_loss_epoch/num_iters_train, total_psnr_epoch/num_iters_train, total_cos_loss_epoch/num_iters_train))
        scheduler.step()
        print("AVG CosLoss: %.05f" % (total_cos_loss_epoch/num_iters_train))
        
        with torch.no_grad():
            student_model.eval()
            total_psnr = 0
            val_loss = 0
            for imgs, _ in tqdm(val_loader):
                imgs = imgs.to(device)
                var_est = 0.01       
  
                real = torch.normal(0, math.sqrt(0.5), (val_batch_size, num_antenna_rx, num_antenna_tx))
                imag = torch.normal(0, math.sqrt(0.5), (val_batch_size, num_antenna_rx, num_antenna_tx))
                val_H = real+1j*imag
                val_H = val_H.to(device)  
                
                real = torch.normal(0, math.sqrt(var_est/2), (imgs.shape[0], num_antenna_rx, num_antenna_tx))
                imag = torch.normal(0, math.sqrt(var_est/2), (imgs.shape[0], num_antenna_rx, num_antenna_tx))
                val_H_delta = real+1j*imag
                val_H_delta = val_H_delta.to(device)   
  
                res_recovery = student_model(imgs, val_snr, val_H, val_H_delta)
                # res_recovery = model.aligned_forward(imgs, val_snr, val_H)
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
                torch.save(student_model.state_dict(), save_path)

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
        



