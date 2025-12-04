import torch
from torch import nn

x = torch.rand((5,2))
fc = nn.Linear(2,2048)

h = fc(x)

img = torch.ones((1,1,3,3))

deconv = nn.ConvTranspose2d(1,1,3,stride=2)
with torch.no_grad():
    deconv.weight[:] = 1.
    deconv.bias[:] = 0.
img_upsample = deconv(img)

deconv1 = nn.ConvTranspose2d(1,1,3,stride=2,padding=1)
with torch.no_grad():
    deconv1.weight[:] = 1.
    deconv1.bias[:] = 0.
deconv1(img)


import numpy as np

def conv_transpose2d_single(x, kernel, stride=1, padding=0, output_padding=0, bias=0.0):
    """
    x:       (H_in, W_in)
    kernel:  (K_h, K_w)
    Returns: (H_out, W_out)
    """
    H_in, W_in = x.shape
    K_h, K_w = kernel.shape
    
    # Output size from PyTorch formula
    H_out = (H_in - 1) * stride - 2 * padding + K_h + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + K_w + output_padding
    
    out = np.zeros((H_out, W_out), dtype=np.float32)
    
    # Scatter-add contributions from each input location
    for i in range(H_in):
        for j in range(W_in):
            val = x[i, j]
            if val == 0:
                continue
            for ki in range(K_h):
                for kj in range(K_w):
                    oi = i * stride - padding + ki
                    oj = j * stride - padding + kj
                    if 0 <= oi < H_out and 0 <= oj < W_out:
                        out[oi, oj] += val * kernel[ki, kj]
    
    # Add bias
    out += bias
    return out
