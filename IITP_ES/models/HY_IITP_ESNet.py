import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models.conv_stft import ConvSTFT, ConviSTFT

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles

class cLN(nn.Module):
    def __init__(self, dimension):
        super(cLN, self).__init__()
        self.gain = nn.Parameter(torch.ones(1, dimension, 1))
        self.bias = nn.Parameter(torch.zeros(1, dimension, 1))

    def forward(self, input, eps=1e-8):
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        device = cum_sum.device

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type()).to(device)
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum).to(device)        

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = torch.sqrt(cum_var + eps)  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())

class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        # N x 1 x 1
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return cLN(dim)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)

class HY_IITP_ESNet1(nn.Module):
    def __init__(self,
                 model_options,
                 non_linear="sigmoid"):
        super(HY_IITP_ESNet1, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax
        }        
        self.fix = True if model_options.fix == "True" else False
        self.win_type= model_options.win_type
        self.win_len = model_options.win_len
        self.win_inc = model_options.win_inc
        self.fft_len = model_options.win_len
        N = self.fft_len // 2 + 1
        B = model_options.B
        H = model_options.H
        non_linear = model_options.non_linear
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=self.fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=self.fix)
        self.ln = build_norm(model_options.norm, 2*N)
        self.proj = nn.Conv1d(2*N, B, 1)
        self.ln2 = build_norm(model_options.norm, B)
        self.proj2 = nn.Conv1d(B, B, 3, padding=1)
        self.gru = nn.GRU(input_size=B, hidden_size=H, num_layers=2, bias=True, batch_first=True)
        self.proj_out = nn.Conv1d(H, N, 1)

    def forward(self, x):
        mic = x[0].float()
        far = x[1].float()     
        bat_size = mic.shape[0]
        org_len = mic.shape[1]
        if org_len % self.win_inc != 0:
            pad_len = int(np.ceil(org_len / self.win_inc) * self.win_inc - org_len)
            mic = F.pad(mic, (0, pad_len, 0, 0), "constant", 0)            
            far = F.pad(far, (0, pad_len, 0, 0), "constant", 0)
        mic_specs = self.stft(mic)        
        far_specs = self.stft(far)
        mic_real = mic_specs[:,:self.fft_len//2+1]
        mic_imag = mic_specs[:,self.fft_len//2+1:]
        far_real = far_specs[:,:self.fft_len//2+1]
        far_imag = far_specs[:,self.fft_len//2+1:]
        mic_spec_mags = torch.sqrt(mic_real**2+mic_imag**2+1e-8)        
        mic_spec_phase = torch.atan2(mic_imag, mic_real)
        far_spec_mags = torch.sqrt(far_real**2+far_imag**2+1e-8)
        inputs = torch.cat([mic_spec_mags, far_spec_mags], 1)
        w = self.proj(self.ln(inputs))
        w = self.proj2(self.ln2(w)).transpose(1, 2).contiguous()
        w_out, hn = self.gru(w)
        out_weight = self.non_linear(self.proj_out(w_out.transpose(1, 2).contiguous()))
        out_mag = mic_spec_mags * out_weight
        out_real = out_mag * torch.cos(mic_spec_phase)
        out_imag = out_mag * torch.sin(mic_spec_phase)
        out_spec = torch.cat([out_real, out_imag], 1)
        out_wav = [self.istft(out_spec).reshape(bat_size, org_len, -1)]
        return out_wav