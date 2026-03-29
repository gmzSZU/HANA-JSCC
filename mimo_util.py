import torch

def complex2real(x):
    x_real = torch.real(x)
    x_imag = torch.imag(x)
    out = torch.cat((x_real, x_imag), dim=-1)
    return out

def calculate_noise_sigma(snr_db):
    """
    根据给定的接收信号 SNR (以 dB 为单位)，计算噪声的标准差 sigma。
    
    参数:
    snr_db: 接收信号的信噪比 (SNR), 单位 dB。
    
    返回:
    sigma: 噪声的标准差。
    """
    # 将 SNR 从 dB 转换为线性比例
    snr_linear = 10 ** (snr_db / 10.0)
    
    # 发射信号的功率假设为 1，因此噪声方差为 1 / SNR
    sigma = 1.0 / torch.sqrt(torch.tensor(snr_linear))
    
    return sigma

def power_normalization(x):
    """
    对复信号进行功率归一化。
    
    参数:
    x: 复信号 (大小为 batch_size x C)，
    
    返回:
    x_normalized: 归一化后的复信号
    """
    # 计算信号的功率 (batch_size, 1)
    power = torch.mean(torch.abs(x)**2, dim=1, keepdim=True)  # (batch_size, 1)
    
    # 进行功率归一化
    x_normalized = x / torch.sqrt(power)  # (batch_size, C)

    return x_normalized

def power_normalization_real_imag(x_real, x_imag):
    """
    对拆分为实部和虚部的信号进行功率归一化。
    
    参数:
    x_real: 复数信号的实部 (batch_size, C)
    x_imag: 复数信号的虚部 (batch_size, C)
    
    返回:
    x_real_normalized: 归一化后的信号实部 (batch_size, C)
    x_imag_normalized: 归一化后的信号虚部 (batch_size, C)
    """
    # 计算信号的功率 (batch_size, 1)
    power = torch.mean(x_real**2 + x_imag**2, dim=1, keepdim=True)  # (batch_size, 1)
    
    # 对实部和虚部分别进行功率归一化
    x_real_normalized = x_real / torch.sqrt(power)  # 归一化后的实部 (batch_size, C)
    x_imag_normalized = x_imag / torch.sqrt(power)  # 归一化后的虚部 (batch_size, C)
    
    return x_real_normalized, x_imag_normalized

# 生成 Rayleigh 衰落信道矩阵
def generate_rayleigh_channel(num_tx, num_rx=1, batch_size=1):
    """
    生成一个 Rayleigh 衰落信道矩阵。

    参数:
    num_tx: 发送天线数 (Nt)
    num_rx: 接收天线数 (Nr), 对于MISO系统是1
    batch_size: 批处理大小

    返回:
    H: 信道矩阵，大小为 (batch_size, num_rx, num_tx)
    """
    H_real = torch.randn(batch_size, num_rx, num_tx)  # 实部
    H_imag = torch.randn(batch_size, num_rx, num_tx)  # 虚部
    H = (H_real + 1j * H_imag) / torch.sqrt(torch.tensor(2.0))  # Rayleigh 衰落
    return H

# 生成复数高斯噪声
def generate_noise(num_antenna_rx, d, sigma, batch_size=1):
    """
    生成复数高斯噪声矩阵。

    参数:
    num_samples: 噪声样本的数量 (C)
    sigma: 噪声的标准差
    batch_size: 批处理大小

    返回:
    N: 噪声矩阵，大小为 (batch_size, num_samples, 1)
    """
    N_real = torch.randn(batch_size, num_antenna_rx, d) * sigma
    N_imag = torch.randn(batch_size, num_antenna_rx, d) * sigma
    N = N_real + 1j * N_imag
    return N

# 波束成形 (最大比率传输)
def beamforming_mrt_miso(H, x_r, x_i):
    """
    最大比率传输 (MRT) 波束成形。
    
    参数:
    H: 信道矩阵 (大小为 batch_size x Nr x Nt)
    x: 发送信号 (大小为 batch_size x Nt x 1)
    
    返回:
    x_bf: 波束成形后的发送信号 (大小为 batch_size x Nt x 1)
    """
    # x = power_normalization(x)
    H_Hermitian = torch.conj(H).transpose(1, 2)  # H^H (大小为 batch_size x Nt x Nr)
    bf_weight = H_Hermitian / torch.norm(H_Hermitian, dim=1, keepdim=True)  # 最大比率传输 (MRT) 权重
    bf_weight = bf_weight.to(x_r.device)
    bf_weight_r, bf_weight_i = bf_weight.real, bf_weight.imag
    x_bf_r = bf_weight_r * x_r - bf_weight_i * x_i
    x_bf_i = bf_weight_r * x_i + bf_weight_i * x_r
    return x_bf_r, x_bf_i

def beamforming_mrt_mimo(H, x_r, x_i):
    H_Hermitian = torch.conj(H).transpose(1, 2)  # H^H (大小为 batch_size x Nt x Nr)
    bf_weight = H_Hermitian / torch.norm(H_Hermitian, dim=1, keepdim=True)  # 最大比率传输 (MRT) 权重
    bf_weight = bf_weight.to(x_r.device)
    bf_weight_r, bf_weight_i = bf_weight.real, bf_weight.imag
    x_bf_r = torch.bmm(bf_weight_r, x_r) - torch.bmm(bf_weight_i, x_i)
    x_bf_i = torch.bmm(bf_weight_r, x_i) + torch.bmm(bf_weight_i, x_r)
    return x_bf_r, x_bf_i

