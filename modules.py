# =============================================================================
# Digital Semantic Communication System Presented by Shuoyao Wang and Mingze Gong, Shenzhen University. 
# =============================================================================
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
from GDN import GDN
from skimage.metrics import peak_signal_noise_ratio as compute_pnsr
import os
from PIL import Image

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,bias=False)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv=conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn=nn.BatchNorm2d(out_channels)
        self.prelu=nn.PReLU()
        
    def forward(self, x): 
        out=self.conv(x)
        out=self.bn(out)
        out=self.prelu(out)
        return out


class deconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(deconv_block, self).__init__()
        self.deconv=deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  output_padding=output_padding)
        self.bn=nn.BatchNorm2d(out_channels)
        self.prelu=nn.PReLU()
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x, activate_func='prelu'): 
        out=self.deconv(x)
        out=self.bn(out)
        if activate_func=='prelu':
            out=self.prelu(out)
        elif activate_func=='sigmoid':
            out=self.sigmoid(out)
        return out  
    

class conv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1x1=False, kernel_size=3, stride=1, padding=1):
        super(conv_ResBlock, self).__init__()
        self.conv1=conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2=conv(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gdn1=GDN(out_channels)
        self.gdn2=GDN(out_channels)
        self.prelu=nn.PReLU()
        self.use_conv1x1=use_conv1x1
        if use_conv1x1 == True:
            self.conv3=conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            
    def forward(self, x): 
        out=self.conv1(x)
        out=self.gdn1(out)
        out=self.prelu(out)
        out=self.conv2(out)
        out=self.gdn2(out)
        if self.use_conv1x1 == True:
            x=self.conv3(x)
        out=out+x
        out=self.prelu(out)
        return out 

class deconv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(deconv_ResBlock, self).__init__()
        self.deconv1=deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.deconv2=deconv(out_channels, out_channels, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.gdn1=GDN(out_channels)
        self.gdn2=GDN(out_channels)
        self.prelu=nn.PReLU()
        self.sigmoid=nn.Sigmoid()
        self.use_deconv1x1=use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3=deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding)
            
    def forward(self, x, activate_func='prelu'): 
        out=self.deconv1(x)
        out=self.gdn1(out)
        out=self.prelu(out)
        out=self.deconv2(out)
        out=self.gdn2(out)
        if self.use_deconv1x1 == True:
            x=self.deconv3(x)
        out=out+x
        if activate_func=='prelu':
            out=self.prelu(out)
        elif activate_func=='sigmoid':
            out=self.sigmoid(out)
        return out 
    

class SE_block(nn.Module):
    def __init__(self, Nin, Nh, No):
        super(SE_block, self).__init__()
        # This is a SENet-like attention block
        self.fc1 = nn.Linear(Nin, Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # out = F.adaptive_avg_pool2d(x, (1,1)) 
        # out = torch.squeeze(out)
        # out = torch.cat((out, snr), 1)
        mu = F.adaptive_avg_pool2d(x, (1,1)).squeeze(-1).squeeze(-1)
        out = self.fc1(mu)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        out = out*x
        return out

class AF_block(nn.Module):
    def __init__(self, Nin, Nh, No):
        super(AF_block, self).__init__()
        self.fc1 = nn.Linear(Nin+1, Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, snr):
        # out = F.adaptive_avg_pool2d(x, (1,1)) 
        # out = torch.squeeze(out)
        # out = torch.cat((out, snr), 1)
        # if snr.shape[0]>1:
        #     snr = snr.squeeze()
        # snr = snr.unsqueeze(1)  
        mu = torch.mean(x, (2, 3))
        out = torch.cat((mu, snr), 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        out = out*x
        return out
   
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

class CustomDataset_Joint_IMG_H(Dataset):
    def __init__(self, image_dataset, channel_data_files, channel_data_folder, transform=None):
        self.image_dataset = image_dataset
        self.channel_data_files = channel_data_files
        self.channel_data_folder = channel_data_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # 获取图像和对应的标签
        image, _ = self.image_dataset[idx]
        # image = self.image_dataset[idx]

        # 随机选择一个信道数据文件
        channel_data_file = np.random.choice(self.channel_data_files)  # 随机选一个 .npy 文件
        channel_data_path = os.path.join(self.channel_data_folder, channel_data_file)

        # 加载信道数据
        channel_info = np.load(channel_data_path)

        # 如果有 transform，将其应用于图像
        if self.transform:
            channel_info = self.transform(channel_info)[0]

        return image, channel_info.to(torch.complex64)
        # channel_info = torch.from_numpy(channel_info)
        # return image, channel_info

class CustomDataset_MIMOMAS_Hhat(Dataset):
    def __init__(self, image_dataset, H_files, H_folder, transform=None):
        self.image_dataset = image_dataset
        
        self.H_files = H_files
        self.H_folder = H_folder
        
        self.transform = transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        # image = self.image_dataset[idx]

        # 随机选择一个信道数据文件
        H_file = np.random.choice(self.H_files)  # 随机选一个 .npy 文件
        H_path = os.path.join(self.H_folder, H_file)
        H = np.load(H_path)
        
        # 如果有 transform，将其应用于图像
        if self.transform:
            H = self.transform(H)[0]

        return image, label, H.to(torch.complex64)


class CustomDataset_MIMOMAS(Dataset):
    def __init__(self, image_dataset, H_files, H_folder, A_rx_files, A_rx_folder, 
                 transform=None):
        self.image_dataset = image_dataset
        
        self.H_files = H_files
        self.H_folder = H_folder
        self.A_rx_files = A_rx_files
        self.A_rx_folder = A_rx_folder
        
        self.transform = transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, _ = self.image_dataset[idx]
        # image = self.image_dataset[idx]

        # 随机选择一个信道数据文件
        H_file = np.random.choice(self.H_files)  # 随机选一个 .npy 文件
        H_path = os.path.join(self.H_folder, H_file)
        H = np.load(H_path)
        
        A_rx_file = np.random.choice(self.A_rx_files)  # 随机选一个 .npy 文件
        A_rx_path = os.path.join(self.A_rx_folder, A_rx_file)
        A_rx = np.load(A_rx_path)
        
        H_hat = np.matmul(A_rx, H)

        # 如果有 transform，将其应用于图像
        if self.transform:
            H = self.transform(H)[0]
            A_rx = self.transform(A_rx)[0]
            H_hat = self.transform(H_hat)[0]

        return image, H.to(torch.complex64), A_rx.to(torch.float32), H_hat.to(torch.complex64)

class CustomDataset_MIMOMAS_fully(Dataset):
    def __init__(self, image_dataset, H_files, H_folder, A_rx_files, A_rx_folder, 
                 A_tx_files, A_tx_folder, transform=None):
        self.image_dataset = image_dataset
        
        self.H_files = H_files
        self.H_folder = H_folder
        self.A_rx_files = A_rx_files
        self.A_rx_folder = A_rx_folder
        self.A_tx_files = A_tx_files
        self.A_tx_folder = A_tx_folder
        
        self.transform = transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, _ = self.image_dataset[idx]
        # image = self.image_dataset[idx]

        # 随机选择一个信道数据文件
        H_file = np.random.choice(self.H_files)  # 随机选一个 .npy 文件
        H_path = os.path.join(self.H_folder, H_file)
        H = np.load(H_path)
        
        A_rx_file = np.random.choice(self.A_rx_files)  # 随机选一个 .npy 文件
        A_rx_path = os.path.join(self.A_rx_folder, A_rx_file)
        A_rx = np.load(A_rx_path)
        
        A_tx_file = np.random.choice(self.A_tx_files)  # 随机选一个 .npy 文件
        A_tx_path = os.path.join(self.A_tx_folder, A_tx_file)
        A_tx = np.load(A_tx_path)

        # 如果有 transform，将其应用于图像
        if self.transform:
            H = self.transform(H)[0]
            A_rx = self.transform(A_rx)[0]
            A_tx = self.transform(A_tx)[0]

        return image, H.to(torch.complex64), A_rx.to(torch.float32), A_tx.to(torch.float32)

def create_mask(N):
    mask = np.zeros((N, N), dtype=np.float32)
    center = N // 2
    mask[center-1, center-1] = 1  # 左上
    mask[center-1, center+1] = 1  # 右上
    mask[center+1, center-1] = 1  # 左下
    mask[center+1, center+1] = 1  # 右下
    mask_flat = mask.flatten()
    one_indices = np.where(mask_flat == 1)[0]
    one_hot_vectors = np.eye(N * N, dtype=np.float32)[one_indices]
    return one_hot_vectors
    
class CustomDataset_MIMOMAS_TcSt(Dataset):
    def __init__(self, image_dataset, H_files, H_folder, H_hat_files, H_hat_folder, 
                 transform=None):
        self.image_dataset = image_dataset
        
        self.H_files = H_files
        self.H_folder = H_folder
        self.H_hat_files = H_hat_files
        self.H_hat_folder = H_hat_folder
        
        self.A_tx = create_mask(N=5)
        
        self.transform = transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, _ = self.image_dataset[idx]
        # image = self.image_dataset[idx]

        # 随机选择一个信道数据文件
        H_file = np.random.choice(self.H_files)  # 随机选一个 .npy 文件
        H_path = os.path.join(self.H_folder, H_file)
        H = np.load(H_path)
        
        H_hat_file = np.random.choice(self.H_hat_files)  # 随机选一个 .npy 文件
        H_hat_path = os.path.join(self.H_hat_folder, H_hat_file)
        H_hat = np.load(H_hat_path)

        # 如果有 transform，将其应用于图像
        if self.transform:
            H = self.transform(H)[0]
            H_hat = self.transform(H_hat)[0]

        return image, H.to(torch.complex64), H_hat.to(torch.complex64)

def cosine_distill_loss(latent_s, latent_t):
    cos_sim = F.cosine_similarity(latent_s, latent_t, dim=1)  # (B,)
    loss = 1 - cos_sim.mean()  # 越小越好
    return loss

def Compute_batch_PSNR(test_input, test_rec):
    psnr_i1 = np.zeros((test_input.shape[0]))
    for j in range(0, test_input.shape[0]):
        psnr_i1[j] = compute_pnsr(test_input[j, :], test_rec[j, :])
    psnr_ave = np.mean(psnr_i1)
    return psnr_ave


def Compute_IMG_PSNR(test_input, test_rec):
    psnr_i1 = np.zeros((test_input.shape[0], 1))
    for j in range(0, test_input.shape[0]):
        psnr_i1[j] = compute_pnsr(test_input[j, :], test_rec[j, :])
    return psnr_i1