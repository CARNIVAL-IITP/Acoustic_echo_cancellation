from utils import T, norm, norm_1d
import torch

def snr(x, s, remove_dc=False):    
    def vec_l2norm(x):
        return torch.norm(x, 2, dim=1)

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
    pred_y = output[0].squeeze()
    true_y = label[0].squeeze()
    loss = -snr(pred_y, true_y)
    return loss