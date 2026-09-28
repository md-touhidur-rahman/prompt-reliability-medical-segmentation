from contextlib import suppress
from functools import partial
import torch


def get_autocast(precision, device_type='cuda'):
    if precision == 'amp':
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return partial(torch.amp.autocast, device_type=device_type)
        return torch.cuda.amp.autocast

    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return partial(torch.amp.autocast, device_type=device_type, dtype=torch.bfloat16)
        return partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)

    else:
        return suppress
