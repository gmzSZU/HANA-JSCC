# =============================================================================
# THIS repo is contributed by Mingze Gong (Graduated Student Member, IEEE), 
# The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China 
# and Shuoyao Wang (Senior Member, IEEE), Shenzhen University, Shenzhen, China
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import encoder_swinvit
import decoder_swinvit
import modules_swinvit
from nn_util import conv_block


class HANA_JSCC(nn.Module):
    def __init__(self, num_antenna_tx, num_antenna_rx):
        super().__init__()  
        # print ("Setting up Virtual MAS Aided MIMO-JSCC...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_antenna_tx = num_antenna_tx
        self.num_antenna_rx = num_antenna_rx
        image_dims = (3, 32, 32)
        self.H = image_dims[1]
        self.W = image_dims[2]
        self.num_downsample = 2
        self.swin_dim = (image_dims[1]//(2**self.num_downsample))**2
        self.mlp_ratio = 4
        self.mlp_hidden = int(self.swin_dim * self.mlp_ratio)
        self.l1_loss = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()
        ## Encoder 
        # Semantic Extraction 
        self.semantic_encoder = encoder_swinvit.SwinJSCC_Encoder(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8],
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True)
        
        # SNR-Adaptation
        self.SA_layer_num = 7
        self.bm_tx_list = nn.ModuleList()
        self.sm_tx_list = nn.ModuleList()
        self.sm_tx_list.append(nn.Linear(256, 384))
        for i in range(self.SA_layer_num):
            if i == self.SA_layer_num - 1:
                outdim = 256
            else:
                outdim = 384
            self.bm_tx_list.append(encoder_swinvit.AdaptiveModulator(384))
            self.sm_tx_list.append(nn.Linear(384, outdim))
            
        # Channel Gain Adaptation (TF Version)        
        self.CGA_num_layers = 6
        # Feature Compression (Phase-I)
        self.channel_encoder_p1 = nn.Linear(1024, 256)       
        # Token Layers
        self.prec_fc = nn.Sequential(nn.Linear(512, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256))
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
                        d_model=256,         
                        nhead=4,             
                        dim_feedforward=1024, 
                        dropout=0.1,         
                        activation='relu',   
                        batch_first=True     
                    )
        self.prec_TF_layers = nn.TransformerEncoder(encoder_layer, num_layers=self.CGA_num_layers)        
        # Projection and Feature Compression (Phase-II)        
        self.channel_encoder_p2 = nn.Sequential(nn.Linear(256, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 32))
        
        

        ## Decoder   
        # Feature Reconstruction (Phase-I)
        self.channel_decoder_p1 = nn.Linear(32, 256)        
        # Channel Gain Adaptation (TF Version)
        # Token Layers
        self.postp_fc = nn.Sequential(nn.Linear(512, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256))
        # Transformer
        decoder_layer = nn.TransformerEncoderLayer(
                        d_model=256,         
                        nhead=4,             
                        dim_feedforward=1024, 
                        dropout=0.1,         
                        activation='relu',   
                        batch_first=True     
                    )
        self.postp_TF_layers = nn.TransformerEncoder(decoder_layer, num_layers=self.CGA_num_layers)        
        # Projection and Feature Reconstruction (Phase-II)
        self.channel_decoder_p2 = nn.Sequential(nn.Linear(256, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 1024))  
            
        # SNR-Adaptation
        self.bm_rx_list = nn.ModuleList()
        self.sm_rx_list = nn.ModuleList()
        self.sm_rx_list.append(nn.Linear(256, 384))
        for i in range(self.SA_layer_num):
            if i == self.SA_layer_num - 1:
                outdim = 256
            else:
                outdim = 384
            self.bm_rx_list.append(encoder_swinvit.AdaptiveModulator(384))
            self.sm_rx_list.append(nn.Linear(384, outdim))
            
        # Semantic Extraction         
        self.semantic_decoder = decoder_swinvit.SwinJSCC_Decoder(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4],
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,) 
    
    def Power_norm(self, s):
        B, C = s.shape[0], s.shape[-1]
        s = s.view(B, -1)
        P_target = 1
        P_current = torch.mean(torch.abs(s)**2, dim=1)
        k = torch.sqrt(torch.div(P_target, (P_current+1e-6)))
        normalized_s = s * k.view(B, -1)        
        return normalized_s.view(B, self.num_antenna_tx, C).to(s.device)
        
    def SNR2std(self, snr):
        snr = 10 ** (snr / 10)
        noise_std = 1 / np.sqrt(2 * snr)
        return noise_std    
    
    def Decomposed_Complex(self, A, B):
        A_real, A_imag = A.real, A.imag
        B_real, B_imag = B.real, B.imag
        AB_real = torch.matmul(A_real, B_real) - torch.matmul(A_imag, B_imag)
        AB_imag = torch.matmul(A_real, B_imag) + torch.matmul(A_imag, B_real)       
        return AB_real+1j*AB_imag
    
    def Encoding(self, x, snr):
        B = x.shape[0]
        device = x.device
        # Semantic Extraction
        x = self.semantic_encoder(x)
        # Channel Adaptation to Additive Noise (From SwinJSCC, thanks)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.SA_layer_num):
            if i == 0:
                temp = self.sm_tx_list[i](x.detach())
            else:
                temp = self.sm_tx_list[i](temp)

            bm = self.bm_tx_list[i](snr_batch).unsqueeze(1).expand(-1, 64, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_tx_list[-1](temp))
        x = x * mod_val        
        # Feature Compression
        x = self.channel_encoder(x.view(B, self.num_antenna_tx, -1))
        return x
    
    def Decoding(self, Rx, snr):
        B = Rx.shape[0]
        device = Rx.device
        # Feature Reconstruction
        Rx = self.channel_decoder(Rx).view(B, 64, 256)
        # Channel Adaptation to Additive Noise (From SwinJSCC, thanks)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.SA_layer_num):
            if i == 0:
                temp = self.sm_rx_list[i](Rx.detach())
            else:
                temp = self.sm_rx_list[i](temp)
            bm = self.bm_rx_list[i](snr_batch).unsqueeze(1).expand(-1, 64, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_rx_list[-1](temp))
        Rx = Rx * mod_val
        # Semantic Decoding
        res = self.semantic_decoder(Rx)
        return res
    
    def forward(self, x, snr, H, H_delta, distill=False):
        # In this MIMO system, CLASSIC MIMO channel is considered. 
        # Assuming the noise power equals to 1.
        B = x.shape[0]
        device = x.device
        H_est = H + H_delta
        # Encoding
        z = self.semantic_encoder(x)
        # Channel Adaptation to Additive Noise (From SwinJSCC, thanks)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.SA_layer_num):
            if i == 0:
                temp = self.sm_tx_list[i](z.detach())
            else:
                temp = self.sm_tx_list[i](temp)

            bm = self.bm_tx_list[i](snr_batch).unsqueeze(1).expand(-1, 64, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_tx_list[-1](temp))
        z = z * mod_val    
        # Feature Compression (Phase-I)
        z = self.channel_encoder_p1(z.view(B, self.num_antenna_tx, -1))
        # Channel Gain Adaptation (TF Version)  
        H_est_4fc = torch.cat((H_est.real.unsqueeze(1), H_est.imag.unsqueeze(1)), dim=-1)
        H_est_4fc = H_est_4fc.view(B, 1, -1)
        csi_token_prec = self.prec_fc(H_est_4fc)
        z = torch.cat((csi_token_prec, z), dim=1)
        z = self.prec_TF_layers(z)
        # Projection and Feature Compression (Phase-II)
        z = self.channel_encoder_p2(z[:, 1:, :])
        # if distill:
        #     buffer = z
        # Power Normalization
        z = z.view(B, self.num_antenna_tx, -1, 2)        
        z = z[:, :, :, 0] + 1j*z[:, :, :, 1]   
        z = self.Power_norm(z)  
        # Imperfect SVD-based Precoding
        U, _, VH = torch.linalg.svd(H_est, full_matrices=False)
        V = torch.conj(VH).transpose(-1, -2)
        Vs = self.Decomposed_Complex(V, z)
        # MIMO Transmission
        y = self.Decomposed_Complex(H, Vs)       
        # Additive Noise Generation
        std = self.SNR2std(snr)
        N = torch.normal(0, std/2, y.shape) + 1j*torch.normal(0, std/2, y.shape)
        N = N.to(device)
        y = y + N
        # Imperfect SVD-based Post-processing
        UH = torch.conj(U).transpose(-1, -2)
        Uy = self.Decomposed_Complex(UH, y)   
        Uy = torch.cat((Uy.real, Uy.imag), dim=-1)
        # Feature Reconstruction (Phase-I)
        Uy = self.channel_decoder_p1(Uy)
        # Channel Gain Adaptation (TF Version)  
        csi_token_postp = self.postp_fc(H_est_4fc)
        Uy = torch.cat((csi_token_postp, Uy), dim=1)
        Uy = self.postp_TF_layers(Uy)
        # Projection and Feature Reconstruction (Phase-II)
        Uy = self.channel_decoder_p2(Uy[:, 1:, :]).view(B, 64, 256)
        if distill:
            buffer_this = Uy
        # Channel Adaptation to Additive Noise (From SwinJSCC, thanks)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.SA_layer_num):
            if i == 0:
                temp = self.sm_rx_list[i](Uy.detach())
            else:
                temp = self.sm_rx_list[i](temp)
            bm = self.bm_rx_list[i](snr_batch).unsqueeze(1).expand(-1, 64, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_rx_list[-1](temp))
        Uy = Uy * mod_val
        # Semantic Decoding
        res = self.semantic_decoder(Uy)
        if distill:
            return res, buffer_this
        else:
            return res
        
    
    def aligned_forward(self, x, snr, H, distill=False):
        # In this MIMO system, CLASSIC MIMO channel is considered. 
        # Assuming the noise power equals to 1.
        B = x.shape[0]
        device = x.device
        # Encoding
        z = self.semantic_encoder(x)
        # Channel Adaptation to Additive Noise (From SwinJSCC, thanks)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.SA_layer_num):
            if i == 0:
                temp = self.sm_tx_list[i](z.detach())
            else:
                temp = self.sm_tx_list[i](temp)

            bm = self.bm_tx_list[i](snr_batch).unsqueeze(1).expand(-1, 64, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_tx_list[-1](temp))
        z = z * mod_val    
        # Feature Compression (Phase-I)
        z = self.channel_encoder_p1(z.view(B, self.num_antenna_tx, -1))
        # Channel Gain Adaptation (TF Version)  
        H_est_4fc = torch.cat((H.real.unsqueeze(1), H.imag.unsqueeze(1)), dim=-1)
        H_est_4fc = H_est_4fc.view(B, 1, -1)
        csi_token_prec = self.prec_fc(H_est_4fc)
        z = torch.cat((csi_token_prec, z), dim=1)
        z = self.prec_TF_layers(z)
        # Projection and Feature Compression (Phase-II)
        z = self.channel_encoder_p2(z[:, 1:, :])
        # if distill:
        #     return z
        # Power Normalization
        z = z.view(B, self.num_antenna_tx, -1, 2)        
        z = z[:, :, :, 0] + 1j*z[:, :, :, 1]   
        z = self.Power_norm(z)  
        # Imperfect SVD-based Precoding
        U, _, VH = torch.linalg.svd(H, full_matrices=False)
        V = torch.conj(VH).transpose(-1, -2)
        Vs = self.Decomposed_Complex(V, z)
        # MIMO Transmission
        y = self.Decomposed_Complex(H, Vs)       
        # Additive Noise Generation
        std = self.SNR2std(snr)
        N = torch.normal(0, std/2, y.shape) + 1j*torch.normal(0, std/2, y.shape)
        N = N.to(device)
        y = y + N
        # Imperfect SVD-based Post-processing
        UH = torch.conj(U).transpose(-1, -2)
        Uy = self.Decomposed_Complex(UH, y)   
        Uy = torch.cat((Uy.real, Uy.imag), dim=-1)
        # Feature Reconstruction (Phase-I)
        Uy = self.channel_decoder_p1(Uy)
        # Channel Gain Adaptation (TF Version)  
        csi_token_postp = self.postp_fc(H_est_4fc)
        Uy = torch.cat((csi_token_postp, Uy), dim=1)
        Uy = self.postp_TF_layers(Uy)
        # Projection and Feature Reconstruction (Phase-II)
        Uy = self.channel_decoder_p2(Uy[:, 1:, :]).view(B, 64, 256)
        if distill:
            return Uy
        # Channel Adaptation to Additive Noise (From SwinJSCC, thanks)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.SA_layer_num):
            if i == 0:
                temp = self.sm_rx_list[i](Uy.detach())
            else:
                temp = self.sm_rx_list[i](temp)
            bm = self.bm_rx_list[i](snr_batch).unsqueeze(1).expand(-1, 64, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_rx_list[-1](temp))
        Uy = Uy * mod_val
        # Semantic Decoding
        res = self.semantic_decoder(Uy)
        
        return res
    
    def forward_CMAwoH(self, x, snr, H, H_delta, distill=False):
        # In this MIMO system, CLASSIC MIMO channel is considered. 
        # Assuming the noise power equals to 1.
        B = x.shape[0]
        device = x.device
        H_est = H + H_delta
        # Encoding
        z = self.semantic_encoder(x)
        # Channel Adaptation to Additive Noise (From SwinJSCC, thanks)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.SA_layer_num):
            if i == 0:
                temp = self.sm_tx_list[i](z.detach())
            else:
                temp = self.sm_tx_list[i](temp)

            bm = self.bm_tx_list[i](snr_batch).unsqueeze(1).expand(-1, 64, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_tx_list[-1](temp))
        z = z * mod_val    
        # Feature Compression (Phase-I)
        z = self.channel_encoder_p1(z.view(B, self.num_antenna_tx, -1))
        # Channel Gain Adaptation (TF Version)  
        z = self.prec_TF_layers(z)
        # Projection and Feature Compression (Phase-II)
        z = self.channel_encoder_p2(z)
        # if distill:
        #     buffer = z
        # Power Normalization
        z = z.view(B, self.num_antenna_tx, -1, 2)        
        z = z[:, :, :, 0] + 1j*z[:, :, :, 1]   
        z = self.Power_norm(z)  
        # Imperfect SVD-based Precoding
        U, _, VH = torch.linalg.svd(H_est, full_matrices=False)
        V = torch.conj(VH).transpose(-1, -2)
        Vs = self.Decomposed_Complex(V, z)
        # MIMO Transmission
        y = self.Decomposed_Complex(H, Vs)       
        # Additive Noise Generation
        std = self.SNR2std(snr)
        N = torch.normal(0, std/2, y.shape) + 1j*torch.normal(0, std/2, y.shape)
        N = N.to(device)
        y = y + N
        # Imperfect SVD-based Post-processing
        UH = torch.conj(U).transpose(-1, -2)
        Uy = self.Decomposed_Complex(UH, y)   
        Uy = torch.cat((Uy.real, Uy.imag), dim=-1)
        # Feature Reconstruction (Phase-I)
        Uy = self.channel_decoder_p1(Uy)
        # Channel Gain Adaptation (TF Version)  
        Uy = self.postp_TF_layers(Uy)
        # Projection and Feature Reconstruction (Phase-II)
        Uy = self.channel_decoder_p2(Uy).view(B, 64, 256)
        if distill:
            buffer_this = Uy
        # Channel Adaptation to Additive Noise (From SwinJSCC, thanks)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.SA_layer_num):
            if i == 0:
                temp = self.sm_rx_list[i](Uy.detach())
            else:
                temp = self.sm_rx_list[i](temp)
            bm = self.bm_rx_list[i](snr_batch).unsqueeze(1).expand(-1, 64, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_rx_list[-1](temp))
        Uy = Uy * mod_val
        # Semantic Decoding
        res = self.semantic_decoder(Uy)
        if distill:
            return res, buffer_this
        else:
            return res


# num_antenna_tx, num_antenna_rx = 16, 16
# model = MIMO_JSCC(num_antenna_tx, num_antenna_rx)

# img = torch.randn(2, 3, 32, 32)
# snr = 10
# real = torch.normal(0, 0.5, (2, num_antenna_rx, num_antenna_tx))
# imag = torch.normal(0, 0.5, (2, num_antenna_rx, num_antenna_tx))
# H = real+1j*imag

# real = torch.normal(0, 0.5, (2, num_antenna_rx, num_antenna_tx))
# imag = torch.normal(0, 0.5, (2, num_antenna_rx, num_antenna_tx))
# H_delta = real+1j*imag

# # res = model(img, snr, H, H_delta)
# res = model.forward_CMAwoH(img, snr, H, H_delta)





