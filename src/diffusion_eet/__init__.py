import math

import torch


def deet(
    low_noise: torch.Tensor,
    high_noise: torch.Tensor,
    deviation: float = 1,
    power: float = 1,
    invert: bool = False,
) -> torch.Tensor:
    if deviation > 0:
        # noise values < deviation will get masked
        mask = low_noise.std() * (math.sqrt(abs(deviation) + 1) - 1) - abs(low_noise)
    elif deviation < 0:
        # noise values > deviation will get masked
        mask = abs(low_noise) - low_noise.std() / (math.sqrt(abs(deviation) + 1) - 1)
    else:
        return low_noise

    mask = (mask**power).clamp(0, 1)
    return low_noise * (1 - mask) + high_noise * mask * (invert * -1 | 1)


def deet_upcast(
    low_noise: torch.Tensor,
    high_noise: torch.Tensor,
    deviation: float = 1,
    power: float = 1,
    invert: bool = False,
    *,
    compute_scale: torch.dtype = torch.float64,
) -> torch.Tensor:
    return deet(
        low_noise.to(dtype=compute_scale),
        high_noise.to(dtype=compute_scale),
        deviation,
        power,
        invert,
    ).to(dtype=low_noise.dtype)
