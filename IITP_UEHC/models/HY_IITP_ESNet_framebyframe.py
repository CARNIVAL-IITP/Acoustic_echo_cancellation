import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models.conv_stft_framebyframe import ConvSTFT, ConviSTFT

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

class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x

class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """
    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x

class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """
    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad      # dcpnv_pad: 2

    def forward(self, x, freq_buf):
        y = self.conv1x1(x)             # x.shape, y.shape, freq_buf.shape: [1, 128, 1], [1, 256, 1], [1, 256, 5]
        y = self.lnorm1(self.prelu1(y)) # [1, 256, 1]
        freq_buf = torch.roll(freq_buf, -1, [2])
        freq_buf[:,:,self.dconv_pad:] = y
        y = self.dconv(freq_buf)        # x.shape, y.shape: [1, 128, 1], [1, 256, 1]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)               # y.shape: [1, 128, 1]
        x = x + y
        return x, freq_buf

class HY_IITP_ESNet1_framebyframe(nn.Module):
    def __init__(self,
                 model_options,
                 non_linear="sigmoid"):
        super(HY_IITP_ESNet1_framebyframe, self).__init__()
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

class HY_IITP_ESNet2_framebyframe(nn.Module):
    def __init__(self,
                 model_options,
                 non_linear="sigmoid"):
        super(HY_IITP_ESNet2_framebyframe, self).__init__()
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
        stack = model_options.X
        repeat = model_options.R
        non_linear = model_options.non_linear
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=self.fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=self.fix)
        self.ln = build_norm(model_options.norm, N)
        self.proj = nn.Conv1d(N, B, 1)
        self.TCN = nn.ModuleList([])
        self.gamma = nn.ModuleList([])
        self.beta = nn.ModuleList([])   
        self.freq_buf = []     
        for rr in range(repeat):
            for ss in range(stack):
                self.TCN.append(Conv1DBlock(B, H, norm="BN", dilation=2**ss, causal = True))
                self.gamma.append(nn.Conv1d(N, B, 1))
                self.beta.append(nn.Conv1d(N, B, 1))
                # we need to update only the first frame so not ( ((model_options.P - 1)**ss)-1 )
                self.freq_buf.append( torch.zeros( 1, 256, 2**ss * (model_options.P - 1)+1 ) )
        self.gamma.append(nn.Conv1d(N, B, 1))
        self.beta.append(nn.Conv1d(N, B, 1))
        self.LSTM = nn.LSTM(input_size=B, hidden_size=H, num_layers=2, bias=True, batch_first=True)
        self.proj_out = nn.Conv1d(H, N, 1)        

    def forward(self, x, hn, cn, past_spec):
        mic = x[0].float()
        far = x[1].float()
        device = mic.device
        bat_size = mic.shape[0]
        org_len = mic.shape[1]
        if org_len % self.win_inc != 0:
            pad_len = int(np.ceil(org_len / self.win_inc) * self.win_inc - org_len)
            mic = F.pad(mic, (0, pad_len, 0, 0), "constant", 0).to(device)
            far = F.pad(far, (0, pad_len, 0, 0), "constant", 0).to(device)
        mic_specs = self.stft(mic)
        far_specs = self.stft(far)
        mic_real = mic_specs[:,:self.fft_len//2+1]
        mic_imag = mic_specs[:,self.fft_len//2+1:]
        far_real = far_specs[:,:self.fft_len//2+1]
        far_imag = far_specs[:,self.fft_len//2+1:]
        mic_spec_mags = torch.sqrt(mic_real**2+mic_imag**2+1e-8)        
        mic_spec_phase = torch.atan2(mic_imag, mic_real)
        far_spec_mags = torch.sqrt(far_real**2+far_imag**2+1e-8)
        inputs = mic_spec_mags          # inputs.shape: [1,129,1]
        w = self.proj(self.ln(inputs))  # w.shape: [1,128,1]
        for i in range(len(self.TCN)):
            tmp_gamma = self.gamma[i](far_spec_mags)
            tmp_beta = self.beta[i](far_spec_mags)
            w = tmp_gamma * w + tmp_beta
            w, self.freq_buf[i] = self.TCN[i](w, self.freq_buf[i])          # w.shape: [1,128,1]
        tmp_gamma = self.gamma[i+1](far_spec_mags)
        tmp_beta = self.beta[i+1](far_spec_mags)
        w = tmp_gamma * w + tmp_beta
        w = w.transpose(1, 2).contiguous()
        w_out, (hn, cn) = self.LSTM(w, (hn,cn))
        out_weight = self.non_linear(self.proj_out(w_out.transpose(1, 2).contiguous()))
        out_mag = mic_spec_mags * out_weight
        out_real = out_mag * torch.cos(mic_spec_phase).to(device)
        out_imag = out_mag * torch.sin(mic_spec_phase).to(device)
        out_spec = torch.cat([out_real, out_imag], 1).to(device)
        out_wav = self.istft(torch.cat( (past_spec, out_spec), 2 ))[0]
        return out_wav, hn, cn, out_spec
    
class HY_IITP_ESNet3_framebyframe(nn.Module):
    def __init__(self,
                model_options,
                non_linear="sigmoid"):
        super(HY_IITP_ESNet3_framebyframe, self).__init__()
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
        self.chan = model_options.chan
        self.N = self.fft_len // 2 + 1
        B = model_options.B
        H = model_options.H
        stack = model_options.X
        repeat = model_options.R
        non_linear = model_options.non_linear
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                            format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=self.fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=self.fix)
        self.ln = build_norm(model_options.norm, self.N * self.chan)
        self.proj = nn.Conv1d(self.N * self.chan, B, 1)
        self.TCN = nn.ModuleList([])
        self.gamma = nn.ModuleList([])
        self.beta = nn.ModuleList([])        
        self.freq_buf = []
        for rr in range(repeat):
            for ss in range(stack):
                self.TCN.append(Conv1DBlock(B, H, norm="BN", dilation=2**ss, causal = True))
                self.gamma.append(nn.Conv1d(self.N, B, 1))
                self.beta.append(nn.Conv1d(self.N, B, 1))
                self.freq_buf.append( torch.zeros( 1, 256, 2**ss * (model_options.P - 1)+1 ) )
        self.gamma.append(nn.Conv1d(self.N, B, 1))
        self.beta.append(nn.Conv1d(self.N, B, 1))
        self.LSTM = nn.LSTM(input_size=B, hidden_size=H, num_layers=2, bias=True, batch_first=True)
        self.proj_out = nn.Conv1d(H, self.N * self.chan, 1)        

    def forward(self, mic, far, hn, cn, past_spec):
        mic = mic.float()     # torch.size [4, 256]
        far = far.float()     # torch.size [1, 256]
        
        # Batch is 1 -> Regard channel as batch
        mic_specs = self.stft(mic)                                  # [chan, 258, 1]
        mic_real = mic_specs[:,:self.fft_len//2+1]                  # [chan, 129, 1]
        mic_imag = mic_specs[:,self.fft_len//2+1:]                  # [chan, 129, 1]
        mic_spec_mags = torch.sqrt(mic_real**2+mic_imag**2+1e-8)    # [chan, 129, 1]
        mic_spec_phase = torch.atan2(mic_imag, mic_real)            # [chan, 129, 1]
        inputs = mic_spec_mags.reshape(1, self.chan * self.N, 1)    # [1, chan * 129, 1]
        
        far_specs = self.stft(far)                                  # [B, 258, 1001]
        far_real = far_specs[:,:self.fft_len//2+1]
        far_imag = far_specs[:,self.fft_len//2+1:]
        far_spec_mags = torch.sqrt(far_real**2+far_imag**2+1e-8)    # [B, 129, 1001]
        
        w = self.proj(self.ln(inputs))                              # w.shape: [1, 128, 1]
        for i in range(len(self.TCN)):
            tmp_gamma = self.gamma[i](far_spec_mags)
            tmp_beta = self.beta[i](far_spec_mags)
            w = tmp_gamma * w + tmp_beta
            w, self.freq_buf[i] = self.TCN[i](w, self.freq_buf[i])
        tmp_gamma = self.gamma[i+1](far_spec_mags)
        tmp_beta = self.beta[i+1](far_spec_mags)
        
        w = tmp_gamma * w + tmp_beta
        w = w.transpose(1, 2).contiguous()
        w_out, (hn, cn) = self.LSTM(w, (hn, cn))
        out_weight = self.non_linear(self.proj_out(w_out.transpose(1, 2).contiguous()))
        
        out_mag  = inputs * out_weight      # [1, chan * 129, 1]
        # Batch is 1 -> Regard channel as batch
        out_mag = out_mag.reshape(self.chan, self.N, 1)         # [chan, 129, 1]
        out_real = out_mag * torch.cos(mic_spec_phase)          # [chan, 129, 1]
        out_imag = out_mag * torch.sin(mic_spec_phase)          # [chan, 129, 1]
        out_spec = torch.cat([out_real, out_imag], 1)           # [chan, 258, 1]
        out_wav = self.istft( torch.cat([past_spec, out_spec], 2) )[:,0,:]
        return out_wav, hn, cn, out_spec