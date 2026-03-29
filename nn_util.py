# =============================================================================
# THIS repo is contributed by Mingze Gong (Graduated Student Member, IEEE) and 
# Shuoyao Wang (Senior Member, IEEE), Shenzhen University, Shenzhen, China
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from GDN import GDN

def compute_conv_params(input_size, output_size, kernel_size, stride):
    # 计算需要的 padding 来满足给定的输入输出关系
    return math.ceil((stride * (output_size - 1) + kernel_size - input_size) / 2)

def create_conv_layers(Nr, Nt, target_h=2, target_w=2):
    # 假设我们设计的第一层卷积的输出是 Nr1 和 Nt1
    Nr1, Nt1 = 3, 4  # 可以调整 Nr1 和 Nt1 为第一层卷积输出的合理尺寸

    # 计算第一层卷积的 stride, kernel_size, padding
    stride_h1 = Nr // Nr1
    stride_w1 = Nt // Nt1
    kernel_h1 = Nr - (Nr1 - 1) * stride_h1
    kernel_w1 = Nt - (Nt1 - 1) * stride_w1
    padding_h1 = compute_conv_params(Nr, Nr1, kernel_h1, stride_h1)
    padding_w1 = compute_conv_params(Nt, Nt1, kernel_w1, stride_w1)

    # 第一层卷积
    conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(kernel_h1, kernel_w1), 
                      stride=(stride_h1, stride_w1), padding=(padding_h1, padding_w1))

    # 计算第二层卷积的 stride, kernel_size, padding
    stride_h2 = Nr1 // target_h
    stride_w2 = Nt1 // target_w
    kernel_h2 = Nr1 - (target_h - 1) * stride_h2
    kernel_w2 = Nt1 - (target_w - 1) * stride_w2
    padding_h2 = compute_conv_params(Nr1, target_h, kernel_h2, stride_h2)
    padding_w2 = compute_conv_params(Nt1, target_w, kernel_w2, stride_w2)

    # 第二层卷积
    conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_h2, kernel_w2), 
                      stride=(stride_h2, stride_w2), padding=(padding_h2, padding_w2))

    return conv1, conv2

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

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()  
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class CFA_module(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_antenna_tx, num_antenna_rx):
        super().__init__()       
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dim_h = 8*num_antenna_tx*num_antenna_rx
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        # H projection
        self.conv_h = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU())
        self.fc_h = nn.Sequential(nn.Linear(self.dim_h, self.in_dim),
                                  nn.ReLU())
        # self.mlp_h = MLP(in_dim=self.in_dim, hidden_dim=self.hidden_dim)
        # Semantic feature projection
        self.mlp_f = MLP(in_dim=self.in_dim, hidden_dim=self.hidden_dim)
    
    def vectorlization(self, H):
        # H (complex) 
        H_real = torch.real(H).unsqueeze(1)
        H_imag = torch.imag(H).unsqueeze(1)
        res = torch.cat((H_real, H_imag), dim=1)
        return res
    
    def forward(self, x, H):
        # H (complex) 
        H_vec = self.vectorlization(H)
        H_hat = self.conv_h(H_vec).view(H_vec.shape[0], -1)
        H_hat = self.fc_h(H_hat)
        vec_h = H_hat.unsqueeze(-1)
        # vec_h = self.mlp_h(H_hat).unsqueeze(-1)
        # feature
        x_maxpool = self.mlp_f(torch.flatten(self.maxpool(x), 1)).unsqueeze(-1)
        x_avgpool = self.mlp_f(torch.flatten(self.avgpool(x), 1)).unsqueeze(-1)
        x_hat = x_maxpool + x_avgpool
        weight = self.sigmoid(vec_h.unsqueeze(-1)+x_hat.unsqueeze(-1))
        return weight*x

class CFA_module_complex(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_antenna_tx, num_antenna_rx):
        super().__init__()       
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dim_h = 8*num_antenna_tx*num_antenna_rx
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        # Phase-I
        self.conv_h = conv_block(in_channels=2, out_channels=8)
        self.fc_h = nn.Sequential(nn.Linear(self.dim_h, self.in_dim),
                                  nn.ReLU())
        self.mlp_h = MLP(in_dim=self.in_dim, hidden_dim=self.hidden_dim)
        # Semantic feature projection
        self.mlp_f = MLP(in_dim=self.in_dim, hidden_dim=self.hidden_dim)
        
        # Phase-II
        self.conv_h_p2_1, self.conv_h_p2_2 = create_conv_layers(num_antenna_rx, num_antenna_tx)
        self.conv_cat = conv_block(in_channels=3, out_channels=1)
    
    def de_complex(self, H):
        # H (complex) 
        H_real = torch.real(H).unsqueeze(1)
        H_imag = torch.imag(H).unsqueeze(1)
        res = torch.cat((H_real, H_imag), dim=1)
        return res
    
    def attn_phaseI(self, x, H):
        # H (complex) 
        H_dc = self.de_complex(H)
        H_hat = self.conv_h(H_dc).view(H_dc.shape[0], -1)
        H_hat = self.fc_h(H_hat)
        vec_h = self.mlp_h(H_hat).unsqueeze(-1)
        # feature
        x_maxpool = self.mlp_f(torch.flatten(self.maxpool(x), 1)).unsqueeze(-1)
        x_avgpool = self.mlp_f(torch.flatten(self.avgpool(x), 1)).unsqueeze(-1)
        x_hat = x_maxpool + x_avgpool
        weight = self.sigmoid(vec_h.unsqueeze(-1)+x_hat.unsqueeze(-1))
        return weight*x

    def attn_phaseII(self, x, H):
        # H (complex)
        H_dc = self.de_complex(H)
        H_hat = self.conv_h_p2_2(self.conv_h_p2_1(H_dc))
        # feature
        x_maxpool, _ = torch.max(x, dim=1, keepdim=True)
        x_avgpool = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat((x_maxpool, x_avgpool), dim=1)
        xH_cat = torch.cat((x_cat, H_hat), dim=1)
        xH_cat = self.conv_cat(xH_cat)
        weight = self.sigmoid(xH_cat)
        return weight*x
    
    def forward(self, x, H):
        f2 = self.attn_phaseI(x, H)
        f3 = self.attn_phaseII(f2, H)
        return f3

class MISO_channel_estimator(nn.Module):
    def __init__(self, num_antenna):
        super().__init__() 
        self.num_antenna = num_antenna
        self.dim_h = int(num_antenna*2)
        self.loss = nn.MSELoss()
        self.conv = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm1d(2),
                                  nn.ReLU())
        self.fc_rec1 = nn.Sequential(nn.Linear(self.dim_h, self.dim_h),
                                     nn.ReLU())
        self.fc_compress = nn.Sequential(nn.Linear(self.dim_h, self.dim_h),
                                         nn.ReLU())
        self.fc_rec2 = nn.Linear(self.dim_h, self.dim_h)
    
    def vectorlization(self, H):
        # H (complex) 
        H_real = torch.real(H)
        H_imag = torch.imag(H)
        res = torch.cat((H_real, H_imag), dim=1)
        return res
    
    def complex_transform(self, H):
        # H (vector, B*dim_h)
        B, C = H.shape
        H_real = H[:, :C//2].unsqueeze(1)
        H_imag = H[:, C//2:].unsqueeze(1)
        res = H_real + 1j*H_imag
        return res
    
    def beamforming(self, H):
        H_vec = self.vectorlization(H)
        bf_mat = self.conv(H_vec)
        return bf_mat
    
    def get_mse_loss(self, H, H_hat):
        H_vec = self.vectorlization(H)
        H_vec = H_vec.view(H.shape[0], -1)
        return self.loss(H_vec, H_hat)
    
    def forward(self, H):
        bf_mat = self.beamforming(H)
        H_tilde = self.fc_rec1(bf_mat.view(H.shape[0], -1))
        H_hat = self.fc_rec2(self.fc_compress(H_tilde))        
        return H_hat

def UPA_DFT_codebook(azimuth_min=0,azimuth_max=1,elevation_min=0,elevation_max=1,n_azimuth=16,n_elevation=16,n_antenna_azimuth=8,n_antenna_elevation=8,spacing=0.5):
    codebook_all = np.zeros((n_azimuth,n_elevation,n_antenna_azimuth*n_antenna_elevation),dtype=np.complex_)

    azimuths = np.linspace(azimuth_min,azimuth_max,n_azimuth,endpoint=False)
    elevations = np.linspace(elevation_min,elevation_max,n_elevation,endpoint=False)

    a_azimuth = np.tile(azimuths*elevations,(n_antenna_azimuth,1)).T
    a_azimuth = 1j*2*np.pi*a_azimuth
    a_azimuth = a_azimuth * np.tile(np.arange(n_antenna_azimuth),(n_azimuth,1))
    a_azimuth = np.exp(a_azimuth)/np.sqrt(n_antenna_azimuth)  

    a_elevation = np.tile(elevations,(n_antenna_elevation,1)).T
    a_elevation = 1j*2*np.pi*a_elevation
    a_elevation = a_elevation * np.tile(np.arange(n_antenna_elevation),(n_elevation,1))
    a_elevation = np.exp(a_elevation)/np.sqrt(n_antenna_elevation)  
    
    codebook_all = np.kron(a_azimuth,a_elevation)
    return codebook_all

class Joint_Tx_Rx_Analog_Beamformer_DFT_Rx(nn.Module):
    def __init__(self, num_antenna_Tx: int, num_antenna_Rx: int, num_beam_Tx: int, num_beam_Rx: int, noise_power: float, norm_factor: float=1.0) -> None:
        super(Joint_Tx_Rx_Analog_Beamformer_DFT_Rx, self).__init__()
        self.num_antenna_Tx = num_antenna_Tx
        self.num_antenna_Rx = num_antenna_Rx
        self.num_beam_Tx = num_beam_Tx
        self.num_beam_Rx = num_beam_Rx
        self.register_buffer('scale_Tx',torch.sqrt(torch.tensor([num_antenna_Tx]).float()))
        self.register_buffer('scale_Rx',torch.sqrt(torch.tensor([num_antenna_Rx]).float()))
        self.register_buffer('noise_power',torch.tensor([noise_power]).float())
        self.register_buffer('norm_factor',torch.tensor([norm_factor]).float())
        self.Tx_codebook_real = nn.Parameter(torch.Tensor(self.num_antenna_Tx, self.num_beam_Tx)) 
        self.Tx_codebook_imag = nn.Parameter(torch.Tensor(self.num_antenna_Tx, self.num_beam_Tx))
        n_ant_per_dim_Rx = int(np.sqrt(num_antenna_Rx))
        n_beam_per_dim_Rx = int(np.sqrt(num_beam_Rx))
        self.register_buffer('Rx_codebook',torch.from_numpy(UPA_DFT_codebook(n_azimuth=n_beam_per_dim_Rx,n_elevation=n_beam_per_dim_Rx,n_antenna_azimuth=n_ant_per_dim_Rx,n_antenna_elevation=n_ant_per_dim_Rx,spacing=0.5).T).to(torch.cfloat))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.Tx_codebook_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Tx_codebook_imag, a=math.sqrt(5))    
        
    def forward(self, h):
        # h is n_batch x num_antenna_Rx x num_antenna_Tx
        Tx_codebook = self.compute_Tx_codebook() # num_antenna_Tx x num_beam_Tx
        Rx_codebook = self.Rx_codebook # num_antenna_Rx x num_beam_Rx
        y = torch.matmul(h,Tx_codebook) # n_batch x num_antenna_Tx x num_beam_Tx       
        noise_real = torch.normal(0,1, size=y.size()).to(h.device)*torch.sqrt(self.noise_power/2)/self.norm_factor
        noise_imag = torch.normal(0,1, size=y.size()).to(h.device)*torch.sqrt(self.noise_power/2)/self.norm_factor
        noise = torch.complex(noise_real, noise_imag)
        y_s = torch.matmul(Rx_codebook.conj().transpose(0,1),y)
        y_n = torch.matmul(Rx_codebook.conj().transpose(0,1),noise)
        return y_s, y_n
    
    def compute_Tx_codebook(self):
        Tx_codebook = torch.complex(self.Tx_codebook_real,self.Tx_codebook_imag)
        Tx_codebook_normalized = torch.div(Tx_codebook,torch.abs(Tx_codebook))/self.scale_Tx
        return Tx_codebook_normalized
    
    def get_Tx_codebook(self):
        with torch.no_grad():
            Tx_codebook = self.compute_Tx_codebook().clone().detach().numpy()
            return Tx_codebook

    def get_Rx_codebook(self):
        return self.Rx_codebook.numpy()  

class Beam_Predictor_MLP(nn.Module):
    def __init__(self, num_antenna: int):    
        super(Beam_Predictor_MLP, self).__init__()
        self.num_antenna = num_antenna
        self.register_buffer('scale',torch.sqrt(torch.tensor([num_antenna]).float()))
        self.dense1 = nn.LazyLinear(out_features=num_antenna*5)
        self.dense2 = nn.LazyLinear(out_features=num_antenna*5)
        self.dense3 = nn.LazyLinear(out_features=num_antenna*2)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.dense1(x))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        v_real = out[:,:self.num_antenna]
        v_imag = out[:,self.num_antenna:]
        v = torch.complex(v_real,v_imag)
        v = torch.div(v,torch.abs(v))/self.scale
        return v 

class MISO_beamformer(nn.Module):
    def __init__(self, num_antenna_Tx: int, num_antenna_Rx: int, num_probing_beam_Tx: int, num_probing_beam_Rx: int, 
                 theta_Tx = None, theta_Rx = None, noise_power = 0.0, norm_factor = 1.0):
        super().__init__()        
        self.num_antenna_Tx = num_antenna_Tx
        self.num_antenna_Rx = num_antenna_Rx
        self.num_probing_beam_Tx = num_probing_beam_Tx
        self.num_probing_beam_Rx = num_probing_beam_Rx
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)

        self.joint_beamformer = Joint_Tx_Rx_Analog_Beamformer_DFT_Rx(num_antenna_Tx = num_antenna_Tx, num_antenna_Rx = num_antenna_Rx, 
                                                              num_beam_Tx = num_probing_beam_Tx, num_beam_Rx = num_probing_beam_Rx,
                                                             noise_power = self.noise_power, norm_factor = self.norm_factor)            
         
        assert self.num_probing_beam_Tx == self.num_probing_beam_Rx, f"number of Tx and Rx probing beams must be the same to use diagonal measurements, got: {self.num_probing_beam_Tx} x {self.num_probing_beam_Rx}"

        self.Tx_beam_predictor = Beam_Predictor_MLP(num_antenna = num_antenna_Tx)
        self.Rx_beam_predictor = Beam_Predictor_MLP(num_antenna = num_antenna_Rx)     

    def get_probing_codebooks(self):
        Tx_probing_codebook = self.joint_beamformer.get_Tx_codebook()
        Rx_probing_codebook = self.joint_beamformer.get_Rx_codebook()
        return Tx_probing_codebook,Rx_probing_codebook 
    
    def forward(self, x):
        bf_signal_s, bf_signal_n = self.joint_beamformer(x) # n_batch x num_beam_Rx x num_beam_Tx, signal and noise components    
        bf_signal = bf_signal_s + bf_signal_n
        bf_signal_power = torch.pow(torch.abs(bf_signal),2)
        bf_signal_power_noiseless = torch.pow(torch.abs(bf_signal_s),2)

        # use diagonal elements of Y
        bf_signal_power_feedback = torch.diagonal(bf_signal_power,dim1=1,dim2=2) # n_batch x num_beam_Tx(or num_beam_Rx)
        bf_signal_power_feedback_noiseless = torch.diagonal(bf_signal_power_noiseless,dim1=1,dim2=2) # n_batch x num_beam_Tx(or num_beam_Rx)
        bf_signal_power_measured = bf_signal_power_feedback

        Tx_beam = self.Tx_beam_predictor(bf_signal_power_feedback)
        Rx_beam = self.Rx_beam_predictor(bf_signal_power_measured)
        return Tx_beam, Rx_beam, bf_signal_power_feedback_noiseless.squeeze()

