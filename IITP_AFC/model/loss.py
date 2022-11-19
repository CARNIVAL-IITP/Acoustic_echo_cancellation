import torch
from torch.nn.utils.rnn import pad_sequence

def mse_loss():
    def loss_func(est, label, nframes):
        # shape: [Batch, Real/Complex, Time(frames), Frequency]
        eps = 1e-7
        new_est = est
        new_label = label
        with torch.no_grad():
            # make mask
            idx = 0
            for frame_num in nframes:
                new_est[idx,:,frame_num:,:] = torch.zeros(1, est.shape[1], est.shape[2] - frame_num, est.shape[3])
                new_label[idx,:,frame_num:,:] = torch.zeros(1, label.shape[1], label.shape[2] - frame_num, label.shape[3])                
                idx += 1
            # input: list of tensor
        # Not to see padded part
        loss = ((new_est - new_label) ** 2).sum() / (est.shape[1]*est.shape[3]*sum(nframes)) + eps
        return loss
    return loss_func