from .loss_IITP_ESNet import loss_IITP_ESNet, loss_IITP_ESNet_corr
from attrdict import AttrDict

def get_lossfns():
    loss_fns = AttrDict()
    loss_fns["loss_IITP_ESNet"] = loss_IITP_ESNet
    loss_fns["loss_IITP_ESNet_corr"] = loss_IITP_ESNet_corr
    return loss_fns
