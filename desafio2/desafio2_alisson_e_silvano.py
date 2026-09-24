import math
import torch

def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.selu(x)


@torch.no_grad()
def inicializar(W, b, fan_in, fan_out, camada, n_camadas):
    # LeCun normal: variância 1/fan_in (par natural da SELU)
    std = 1.0 / math.sqrt(fan_in)
    if camada == n_camadas:
        std *= 0.5
    W.normal_(0.0, std)
    b.zero_()
