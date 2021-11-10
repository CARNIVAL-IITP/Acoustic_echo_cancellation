from .loss_IITP_ESNet import loss_IITP_ESNet
from attrdict import AttrDict

def get_lossfns():
    loss_fns = AttrDict()
    loss_fns["loss_IITP_ESNet"] = loss_IITP_ESNet
    return loss_fns
