from utils import T, norm, norm_1d
import torch

def snr(x, s, remove_dc=False):    
    def vec_l2norm(x):
        tot = 0
        for chan in range(x.shape[1]):
            tot = tot + torch.norm(x[:, chan, :], 2, dim=1)
        return tot / x.shape[1]

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - torch.mean(x)
        s_zm = s - torch.mean(s)
        t = torch.dot(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = s
        n = x - t
    return 20 * torch.log10(vec_l2norm(t) / (vec_l2norm(n)+1e-8))

def log_mse_loss(ref, est, max_snr=1e6, bias_ref_signal=None):  
  err_pow = torch.sum(torch.square(ref - est), dim=-1)  
  snrfactor = 10.**(-max_snr / 10.)
  if bias_ref_signal is None:
    ref_pow = torch.sum(torch.square(ref), dim=-1)
  else:
    ref_pow = torch.sum(torch.square(bias_ref_signal), dim=-1)
  bias = snrfactor * ref_pow  
  return 10. * torch.log10(bias + err_pow + 1e-8)

def loss_IITP_ESNet(output, label):
    pred_y = output[0].squeeze(-1)
    true_y = label[0].squeeze(-1)
    loss = -snr(pred_y, true_y)
    return loss

def loss_IITP_ESNet_corr(output, label, input, eps=1e-8):    
    
    shat = output[0].squeeze(-1)      # [B, chan, 128000]
    s = label[0].squeeze(-1)          # [B, chan, 128000]
    loss_snr = -snr(shat, s)
    
    # x = input[1].squeeze(-1)          # [B, 128000]
    # y = input[0].squeeze(-1)          # [B, chan, 128000]
    # dhat = y - shat
    # corrxd = torch.sum(x * dhat)**2 / (torch.sum(x **2) * torch.sum(dhat **2) + eps)
    # corrys = torch.sum(y * shat)**2 / (torch.sum(y **2) * torch.sum(shat **2) + eps)
    # corryd = torch.sum(y * dhat)**2 / (torch.sum(y **2) * torch.sum(dhat **2) + eps)
    # return loss_snr, corrxd, corrys, corryd
    return loss_snr
