import importlib
import time
import torch
import numpy as np

def remove_extra_tail(m, size=256):
    assert m.shape[1] >= size, "len(y) should be large than size."
    return m[:, : - (m.shape[1] % size)]

def prepare_empty_dir(dirs, resume=False):
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)

class ExecutionTime:

    def __init__(self):
        self.start_time = time.time()


    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg):
    """
    According to config items, load specific module dynamically with params.

    eg，config items as follow：
        module_cfg = {
            "module": "models.model",
            "main": "Model",
            "args": {...}
        }

    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])
    return getattr(module, module_cfg["main"])(**module_cfg["args"])

def z_score(m):
    mean = torch.mean(m, [1,2])
    std_var = torch.std(m, [1,2])

    # size: [batch] => pad => [batch, T, F]
    mean = mean.expand(m.size()[::-1]).permute(2, 1, 0)
    std_var = std_var.expand(m.size()[::-1]).permute(2, 1, 0)

    return (m - mean) / std_var, mean, std_var

def reverse_z_score(m, mean, std_var):
    return m * std_var + mean

def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min

def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min
