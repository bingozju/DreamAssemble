import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp, biased_softplus
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize



class RNNSmoothSoftmax(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=dim_in, 
            hidden_size=dim_hidden,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.proj = nn.Linear(2*dim_hidden, dim_out)
        
    def forward(self, x):

        # 通过RNN进行平滑
        rnn_out, _ = self.rnn(x)
        smoothed = self.proj(rnn_out)
        
        return smoothed


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

import torch.nn.init as init
def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)
        
# def tempered_softmax(logits, temperature=0.1, dim=-1):
#     logits = logits - logits.max(dim=-1, keepdim=True).values  # 数值稳定
#     exp_logits = torch.exp(logits / temperature)
#     return exp_logits / exp_logits.sum(dim=dim, keepdim=True)

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_smooth_m(tensor: torch.Tensor, kernel_size: int = 9, sigma: float = 0.1) -> torch.Tensor:
    """
    对形状为(N, M, 3)的张量在第二个维度(M)上进行高斯平滑处理。
    
    Args:
        tensor (torch.Tensor): 输入张量，形状为(N, M, 3)。
        kernel_size (int, 可选): 高斯核大小（必须为奇数），默认为5。
        sigma (float, 可选): 高斯核标准差，默认为1.0。
    
    Returns:
        torch.Tensor: 平滑后的张量，形状与输入相同。
    """
    assert kernel_size % 2 == 1, "kernel_size必须是奇数"
    N, M, C = tensor.shape
    device = tensor.device
    dtype = tensor.dtype
    
    # 生成一维高斯核
    x = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) // 2
    y = torch.ones_like(x)
    gauss = torch.exp(-y**2 / (2 * sigma**2))
    gauss /= gauss.sum()  # 归一化
    
    # 调整输入形状：(N, M, 3) → (N, 3, M)
    input_reshaped = tensor.permute(0, 2, 1)  # [N, C, M]
    
    # 创建Conv1d层（每个颜色通道独立处理）
    padding = (kernel_size - 1) // 2
    conv = nn.Conv1d(
        in_channels=C,
        out_channels=C,
        kernel_size=kernel_size,
        padding=padding,
        groups=C,  # 每个通道独立卷积
        bias=False
    ).to(device=device, dtype=dtype)
    
    # 设置卷积核权重（所有通道共享同一高斯核）
    conv.weight.data = gauss.view(1, 1, -1).repeat(C, 1, 1)  # [C, 1, kernel_size]
    conv.weight.requires_grad_(False)
    
    # 应用卷积
    output = conv(input_reshaped)  # 输出形状 [N, C, M]
    
    # 恢复形状：(N, C, M) → (N, M, 3)
    output_reshaped = output.permute(0, 2, 1)
    
    return output_reshaped



        
class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim     
          
        self.part_nums = opt.part_nums
        self.x_centers = nn.Parameter(torch.Tensor(opt.part_centers), requires_grad=True)  #nn.Parameter(torch.Tensor(opt.part_centers), requires_grad=True)
        
        
        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, level_dim=4, log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep')
        

        
        self.sigma_net = MLP(self.in_dim , 3 + self.part_nums, hidden_dim + self.part_nums, num_layers, bias=True)
     
        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else biased_softplus

        # background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=6)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None
        
        for i in range(self.part_nums):
            self.sigma_net.apply(weights_init)
            
    def suppress_output(self, x, gamma=0.9, alpha=10):
        max_val = torch.max(x, dim=-1)
        threshold = gamma * max_val
        mask = torch.sigmoid(alpha * (x - threshold), dim=-1)
        return x * mask

    def common_forward(self, x):

        enc = self.encoder(x, bound=self.bound, max_level=self.max_level)
        h = self.sigma_net(enc)
        
  
 
       
        density_blobs_parts = []
        
        for i in range(len(self.x_centers)):
            density_blobs_parts.append(self.density_blob(x - self.x_centers[i], i)[...,None])

        density_blobs_parts = torch.cat(density_blobs_parts, dim=-1)
        
        albedo = torch.sigmoid(h[...,:3])
        sigma_parts = self.density_activation(h[...,3:] + density_blobs_parts)
      
        sigma_parts = sigma_parts / self.opt.part_nums  #self.suppress_output(sigma_parts)
        
        sigma = torch.sum(sigma_parts, dim=-1)
        
        
        
        return sigma, albedo, sigma_parts, None#softmax
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _, _, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _, _, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _, _, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _, _, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _, _, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _, _, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal

    def normal(self, x):
        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)
        return normal
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        sigma, albedo, sigma_parts, softmax = self.common_forward(x)

        if shading == 'albedo':
            normal = None
            color = albedo
        
        else: # lambertian shading

            # normal = self.normal_net(enc)
            normal = self.normal(x)

            lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
            
        return {
            'sigmas': sigma,
            'rgbs': color,
            'normals':normal,
            'sigma_parts':sigma_parts,
            #'softmax': softmax,
        }

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo, sigma_parts, softmax = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
            'sigma_parts': sigma_parts,
            #'softmax': softmax,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):
        
        params = []

       
        params.append({'params': self.encoder.parameters(), 'lr': lr * 10})
        
        params.append({'params': self.sigma_net.parameters(), 'lr': lr})
             

        if self.opt.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        if self.opt.dmtet and not self.opt.lock_geo:
            params.append({'params': self.sdf, 'lr': lr * 5})
            params.append({'params': self.deform, 'lr': lr * 5})
            for i in range(self.opt.part_nums):
                params.append({'params': getattr(self, f'sdf_{i}'), 'lr': lr * 5})
                params.append({'params': getattr(self, f'deform_{i}'), 'lr': lr * 5})

        return params