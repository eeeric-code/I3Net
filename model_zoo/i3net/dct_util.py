import torch
from torch import nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
# import torch_dct as DCT
# DCT.idct()
# def dct1_torch(x):
#     """
#     Discrete Cosine Transform, Type I
#
#     :param x: the input signal
#     :return: the DCT-I of the signal over the last dimension
#     """
#     x_shape = x.shape
#     x = x.view(-1, x_shape[-1])
#     print(x.shape)
#     print(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1).shape)
#     x = torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), dim=1)
#     print(x.shape)
#     x = x.real
#     print(x.shape)
#     return x.view(*x_shape)
#
#
# def idct1_torch(X):
#     """
#     The inverse of DCT-I, which is just a scaled DCT-I
#
#     Our definition if idct1 is such that idct1(dct1(x)) == x
#
#     :param X: the input signal
#     :return: the inverse DCT-I of the signal over the last dimension
#     """
#     n = X.shape[-1]
#     return dct1_torch(X) / (2 * (n - 1))
def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    # print(v.shape)
    Vc = torch.fft.fft(v, dim=1) # , onesided=False)
    # print(Vc.shape)
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i # [:, :N // 2 + 1]

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    # print(V)
    return V

def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    # print(X)
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    # V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.complex(V_r, V_i)
    v = torch.fft.ifft(V, dim=1) # , onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    x = x.real
    return x.view(*x_shape)


def dct_2d_torch(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d_torch(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def get_dctMatrix(m, n):
    N = n
    C_temp = np.zeros([m, n])
    C_temp[0, :] = 1 * np.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)
                                  ) * np.sqrt(2 / N)
    return torch.tensor(C_temp, dtype=torch.float)


def dct1d(feature, dctMat):
    feature = feature @ dctMat.T # dctMat @ feature  #
    return feature.contiguous()  # torch.tensor(x, device=feature.device)


def idct1d(feature, dctMat):
    feature = feature @ dctMat # .T # dctMat.T @ feature  # .T
    return feature.contiguous()  # torch.tensor(x, device=feature.device)


# def dct2d(feature, dctMat):
#     # print(dctMat.shape, feature.shape)
#     feature = dctMat @ feature
#     # print(dctMat.shape, feature.shape)
#     feature = feature @ dctMat.T
#     return feature.contiguous()  # torch.tensor(x, device=feature.device)
def dct2d(feature, dctMat):
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature, dctMat)# dctMat @ feature
    # print(dctMat.shape, feature.shape)
    # feature = feature @ dctMat.T
    # print(feature.transpose(-1, -2).shape, dctMat.shape)
    feature = dct1d(feature.transpose(-1, -2), dctMat) # dctMat @ feature.transpose(-1, -2) # @ dctMat.T
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)

# def idct2d(feature, dctMat):
#     feature = dctMat.T @ feature
#     feature = feature @ dctMat
#     return feature.contiguous()  # torch.tensor(x, device=feature.device)
def idct2d(feature, dctMat):
    feature = idct1d(feature, dctMat) # dctMat.T @ feature # .transpose(-1, -2)
    feature = idct1d(feature.transpose(-1, -2), dctMat)
    return feature.transpose(-1, -2).contiguous() # torch.tensor(x, device=feature.device)

def dct2dx(feature, dctMat1, dctMat2):
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature, dctMat1) # dctMat1 @ feature
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature.transpose(-1, -2), dctMat2) # feature @ dctMat2.T
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)


def idct2dx(feature, dctMat1, dctMat2):
    feature = idct1d(feature, dctMat1)  # dctMat.T @ feature # .transpose(-1, -2)
    feature = idct1d(feature.transpose(-1, -2), dctMat2)
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)

def dct1_spectral2d_torch(x):
    # b, c, s, h, w = x.shape
    # x = rearrange(x, 'b c s h w -> (b c h w) s')
    x = x.permute(0, 2, 3, 1)
    x = dct(x, 'ortho')
    x = x.permute(0, 3, 1, 2)
    # rearrange(x, ' (b c h w) s -> b c s h w', c=c, h=h, w=w)
    return x.contiguous() # torch.tensor(x, device=feature.device)

def idct1_spectral2d_torch(x):
    # n = feature.shape[-1]
    # b, c, s, h, w = x.shape
    # x = rearrange(x, 'b c s h w -> (b c h w) s')
    x = x.permute(0, 2, 3, 1)
    x = idct(x, 'ortho')
    x = x.permute(0, 3, 1, 2)
    # rearrange(x, ' (b c h w) s -> b c s h w', c=c, h=h, w=w)
    return x.contiguous()
# def dct1_spectral2d(x, dctMat):
#     # b, c, s, h, w = x.shape
#     # x = rearrange(x, 'b c s h w -> (b c h w) s')
#     x = x.permute(0, 2, 3, 1)
#     x = x @ dctMat.T
#     x = x.permute(0, 3, 1, 2)
#     # rearrange(x, ' (b c h w) s -> b c s h w', c=c, h=h, w=w)
#     return x.contiguous() # torch.tensor(x, device=feature.device)
#
# def idct1_spectral2d(x, dctMat):
#     # n = feature.shape[-1]
#     # b, c, s, h, w = x.shape
#     # x = rearrange(x, 'b c s h w -> (b c h w) s')
#     x = x.permute(0, 2, 3, 1)
#     x = x @ dctMat
#     x = x.permute(0, 3, 1, 2)
#     # rearrange(x, ' (b c h w) s -> b c s h w', c=c, h=h, w=w)
#     return x.contiguous()

class SDCTx(nn.Module):
    def __init__(self, heads=1):
        super().__init__()
        self.dctMat = None
        self.heads = heads
    def check_dct_matrix(self, d):
        if self.dctMat is None or d != self.dctMat.shape[-1]:
            self.dctMat = get_dctMatrix(d, d)
    def forward(self, x):

        if self.heads > 1:
            x = rearrange(x, 'b (head c) h w -> b head c h w', head=self.heads)
        self.check_dct_matrix(x.shape[-3])
        self.dctMat = self.dctMat.to(x.device)
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = dct1d(x, self.dctMat)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = dct1d(x, self.dctMat)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if self.heads > 1:
            x = rearrange(x, 'b head c h w -> b (head c) h w')
        return x

class ISDCTx(nn.Module):
    def __init__(self, heads=1):
        super().__init__()
        self.heads = heads
        self.dctMat = None
    def check_dct_matrix(self, d):
        if self.dctMat is None or d != self.dctMat.shape[-1]:
            self.dctMat = get_dctMatrix(d, d)
    def forward(self, x):
        # self.dctMat = self.dctMat.to(x.device)
        if self.heads > 1:
            x = rearrange(x, 'b (head c) h w -> b head c h w', head=self.heads)
        self.check_dct_matrix(x.shape[-3])
        self.dctMat = self.dctMat.to(x.device)
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = idct1d(x, self.dctMat)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = idct1d(x, self.dctMat)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if self.heads > 1:
            x = rearrange(x, 'b head c h w -> b (head c) h w')
        return x
class SDCT(nn.Module):
    def __init__(self, window_size=64, dynamic=False, heads=1):
        super().__init__()
        if not dynamic:
            self.dctMat = get_dctMatrix(window_size//heads, window_size//heads)
        else:
            self.dctMat = nn.Parameter(get_dctMatrix(window_size//heads, window_size//heads),
                                   requires_grad=True)
        self.heads = heads
    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        if self.heads > 1:
            x = rearrange(x, 'b (head c) h w -> b head c h w', head=self.heads)
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = dct1d(x, self.dctMat)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = dct1d(x, self.dctMat)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if self.heads > 1:
            x = rearrange(x, 'b head c h w -> b (head c) h w')
        return x

class ISDCT(nn.Module):
    def __init__(self, window_size=64, dynamic=False, heads=1):
        super().__init__()
        self.heads = heads
        if not dynamic:
            self.dctMat = get_dctMatrix(window_size//heads, window_size//heads)
        else:
            self.dctMat = nn.Parameter(get_dctMatrix(window_size//heads, window_size//heads),
                                   requires_grad=True)

    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        if self.heads > 1:
            x = rearrange(x, 'b (head c) h w -> b head c h w', head=self.heads)
            # x = rearrange(x, 'b (c head) h w -> b head c h w', head=self.heads)
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = idct1d(x, self.dctMat)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = idct1d(x, self.dctMat)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if self.heads > 1:
            x = rearrange(x, 'b head c h w -> b (head c) h w')
            # x = rearrange(x, 'b head c h w -> b (c head) h w')
        return x
class DCT1d(nn.Module):
    def __init__(self, window_size=64):
        super(DCT1d, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)

    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = dct1d(x, self.dctMat)
        return x


class IDCT1d(nn.Module):
    def __init__(self, window_size=64):
        super(IDCT1d, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)

    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        x = idct1d(x, self.dctMat)
        return x
class DCT1x(nn.Module):
    def __init__(self, dim=-1):
        super(DCT1x, self).__init__()
        self.dctMat = None
        self.dim = dim

    def check_dct_matrix(self, d):
        if self.dctMat is None or d != self.dctMat.shape[-1]:
            self.dctMat = get_dctMatrix(d, d)

    def forward(self, x):
        if self.dim != -1 or self.dim != len(x.shape)-1:
            x = x.transpose(self.dim, -1)
        self.check_dct_matrix(x.shape[-1])

        self.dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = dct1d(x, self.dctMat)
        if self.dim != -1 or self.dim != len(x.shape)-1:
            x = x.transpose(self.dim, -1)
        return x.contiguous()


class IDCT1x(nn.Module):
    def __init__(self, dim=-1):
        super(IDCT1x, self).__init__()
        self.dctMat = None
        self.dim = dim
    def check_dct_matrix(self, d):
        if self.dctMat is None or d != self.dctMat.shape[-1]:
            self.dctMat = get_dctMatrix(d, d)
    def forward(self, x):
        if self.dim != -1 or self.dim != len(x.shape) - 1:
            x = x.transpose(self.dim, -1)
        self.check_dct_matrix(x.shape[-1])

        self.dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = idct1d(x, self.dctMat)
        if self.dim != -1 or self.dim != len(x.shape) - 1:
            x = x.transpose(self.dim, -1)
        return x.contiguous()
class DCT2(nn.Module):
    def __init__(self, window_size=8, norm='ortho'):
        super(DCT2, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)
        self.norm = norm
        self.window_size = window_size
    def forward(self, x):
        dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = dct2d(x, dctMat)
        return x


class IDCT2(nn.Module):
    def __init__(self, window_size=8, norm='ortho'):
        super(IDCT2, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)
        self.norm = norm
        self.window_size = window_size
    def forward(self, x):
        dctMat = self.dctMat.to(x.device)
        x = idct2d(x, dctMat)
        return x
class RFFT2(nn.Module):
    def __init__(self, norm='ortho'):
        super(RFFT2, self).__init__()
        self.norm = norm

    def forward(self, x):
        x = torch.fft.rfft2(x, norm=self.norm)

        return torch.cat([x.real, x.imag], dim=1)


class IRFFT2(nn.Module):
    def __init__(self, norm='ortho'):
        super(IRFFT2, self).__init__()
        self.norm = norm

    def forward(self, x):
        x_real, x_imag = x.chunk(2, dim=1)
        x = torch.complex(x_real, x_imag)
        x = torch.fft.irfft2(x, norm=self.norm)
        # print(x.shape)
        return x
class DCT2x(nn.Module):
    def __init__(self, norm='ortho'):
        super(DCT2x, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm

    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1] and w != self.dctMatW.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW.shape[-1]:
            self.dctMatW = get_dctMatrix(w, w)

    def forward(self, x):
        _, _, h, w = x.shape
        self.check_dct_matrix(h, w)
        dctMatH = self.dctMatH.to(x.device)
        dctMatW = self.dctMatW.to(x.device)
        # print(x.shape, self.dctMatH.shape, self.dctMatW.shape)
        x = dct2dx(x, dctMatW, dctMatH)

        return x


class IDCT2x(nn.Module):
    def __init__(self, norm='ortho'):
        super(IDCT2x, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm

    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1] and w != self.dctMatW.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW.shape[-1]:
            self.dctMatW = get_dctMatrix(w, w)

    def forward(self, x):
        _, _, h, w = x.shape
        self.check_dct_matrix(h, w)
        dctMatH = self.dctMatH.to(x.device)
        dctMatW = self.dctMatW.to(x.device)
        x = idct2dx(x, dctMatW, dctMatH)

        return x


class DCT2d_fold_branch(nn.Module):
    def __init__(self, window_size=8, pad_size=0, stride=1, pad_mode='reflect', dct_torch=False):
        super().__init__()
        self.window_size = window_size
        # n = window_size ** 2
        self.dct_torch = dct_torch
        if not dct_torch:
            self.dct2d = DCT2(window_size=window_size, norm='ortho')
            self.idct2d = IDCT2(window_size=window_size, norm='ortho')
        self.dct_matrix = get_dctMatrix(window_size, window_size)
        self.stride = stride
        self.mode = pad_mode
        if self.mode != 'reflect':
            self.pad_size = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=self.pad_size // 2, stride=self.stride)
        else:
            pad_size = window_size - 1
            self.pad_size = (pad_size//2, pad_size - pad_size//2)
            self.pad_sizex = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=self.stride)
        output_size = [1, 1, 128, 128]
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)

        self.input_ones = torch.ones(output_size)
        self.divisor = self.fold(self.unfold(self.input_ones))

    def get_bound(self, x):
        self.dct_matrix = self.dct_matrix.to(x.device)
        bound = rearrange(self.dct_matrix, 'h w -> (h w)')
        return bound / 2.
    def dct_forward(self, x):
        b, self.c, H, W = x.shape

        # _, _, H, W = x.shape
        x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
        if self.mode == 'reflect':
            x = F.pad(x, (self.pad_size[0], self.pad_size[1], self.pad_size[0], self.pad_size[1]), mode=self.mode)
        self.shape_x = x.shape
        # print('y: ', x.shape)
        x = self.unfold(x)
        # print('unfold: ', x.shape)

        if self.mode != 'reflect':
            self.h, self.w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
                    W + 2 * self.pad_size - self.window_size) // self.stride + 1
        else:
            self.h, self.w = (H + self.pad_sizex - self.window_size) // self.stride + 1, (
                    W + self.pad_sizex - self.window_size) // self.stride + 1
        x = rearrange(x, 'b (h w) n -> b n h w', h=self.window_size, w=self.window_size)
        if not self.dct_torch:
            x = self.dct2d(x)
        else:
            # print(x.max(), x.min())
            x = dct_2d_torch(x, 'ortho')
            # print(x.max(), x.min())
        return rearrange(x, 'b (k1 k2) h w -> b (h w) k1 k2', k1=self.h, k2=self.w)
    def idct_forward(self, x):
        if self.shape_x[-2:] != self.divisor.shape[-2:]:
            h, w = self.shape_x[-2:]
            self.input_ones = torch.ones([1, 1, h, w])
            self.input_ones = self.input_ones.to(x.device)
            self.fold = nn.Fold(output_size=self.shape_x[-2:], **self.fold_params)
            self.divisor = self.fold(self.unfold(self.input_ones))
        if self.divisor.device != x.device:
            self.divisor = self.divisor.to(x.device)
        # x = rearrange(x, 'b c h w -> b c (h w)', h=self.h, w=self.w)
        x = rearrange(x, 'b (h w) k1 k2 -> b (k1 k2) h w', h=self.window_size, w=self.window_size)
        # x = self.idct2d(x)
        if not self.dct_torch:
            x = self.idct2d(x)
        else:
            x = idct_2d_torch(x, 'ortho')
        x = rearrange(x, 'b n h w -> b (h w) n')

        x = self.fold(x) / self.divisor
        # print('fold: ', x.shape)
        if self.mode == 'reflect':
            x = x[:, :, self.pad_size[0]:-self.pad_size[1], self.pad_size[0]:-self.pad_size[1]].contiguous()
        x = rearrange(x, '(b c) k h w -> b (c k) h w', c=self.c, k=1)
        # out = self.project_out(x)
        return x
    
    def forward(self, x, dct_forward=True):
        if dct_forward:
            return self.dct_forward(x)
        else:
            return self.idct_forward(x)
        # b, c, H, W = x.shape
        # # _, _, H, W = x.shape
        # x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
        # if self.mode == 'reflect':
        #     x = F.pad(x, (self.pad_size, 0, self.pad_size, 0), mode=self.mode)
        # if x.shape[-2:] != self.divisor.shape[-2:]:
        #     self.input_ones = torch.ones_like(x)
        #     self.fold = nn.Fold(output_size=x.shape[-2:], **self.fold_params)
        #     self.divisor = self.fold(self.unfold(self.input_ones))
        # if self.divisor.device != x.device:
        #     self.divisor = self.divisor.to(x.device)
        # # print('y: ', x.shape)
        # x = self.unfold(x)
        # # print('unfold: ', x.shape)
        #
        # if self.mode != 'reflect':
        #     h, w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
        #                 W + 2 * self.pad_size - self.window_size) // self.stride + 1
        # else:
        #     h, w = (H + self.pad_size - self.window_size) // self.stride + 1, (
        #             W + self.pad_size - self.window_size) // self.stride + 1
        # x = rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
        # x = self.dct2d(x)
        # # x = self.conv2(self.conv1(x))
        # # x = self.act(x)
        # # x = self.conv3(x) + short_cut
        # x = self.idct2d(x)
        # x = rearrange(x, 'b c h w -> b c (h w)', h=h, w=w)
        #
        # x = self.fold(x) / self.divisor
        # # print('fold: ', x.shape)
        # if self.mode == 'reflect':
        #     x = x[:, :, self.pad_size:, self.pad_size:]
        # x = rearrange(x, '(b c) k h w -> b (c k) h w', b=b, c=c, k=1)
        # # out = self.project_out(x)
        # return x

class DCT1d_fold_branch(nn.Module):
    def __init__(self, window_size=8, pad_size=0, stride=1, pad_mode='reflect'):
        super().__init__()
        self.window_size = window_size
        n = window_size ** 2
        self.dct1d = DCT1d(window_size=n)
        self.idct1d = IDCT1d(window_size=n)
        self.dct_matrix = get_dctMatrix(n, n)
        self.stride = stride
        self.mode = pad_mode
        if self.mode != 'reflect':
            self.pad_size = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=self.pad_size // 2, stride=self.stride)
        else:
            self.pad_size = window_size - 1
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=self.stride)
        output_size = [1, 1, 128, 128]
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)

        self.input_ones = torch.ones(output_size)
        self.divisor = self.fold(self.unfold(self.input_ones))
    def dct_forward(self, x):
        self.b, self.c, H, W = x.shape

        # _, _, H, W = x.shape
        x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
        if self.mode == 'reflect':
            x = F.pad(x, (self.pad_size, 0, self.pad_size, 0), mode=self.mode)
        self.shape_x = x.shape
        # if x.shape[-2:] != self.divisor.shape[-2:]:
        #     self.input_ones = torch.ones_like(x)
        #     self.fold = nn.Fold(output_size=x.shape[-2:], **self.fold_params)
        #     self.divisor = self.fold(self.unfold(self.input_ones))
        # if self.divisor.device != x.device:
        #     self.divisor = self.divisor.to(x.device)
        # print('y: ', x.shape)
        x = self.unfold(x)
        # print('unfold: ', x.shape)

        if self.mode != 'reflect':
            self.h, self.w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
                    W + 2 * self.pad_size - self.window_size) // self.stride + 1
        else:
            self.h, self.w = (H + self.pad_size - self.window_size) // self.stride + 1, (
                    W + self.pad_size - self.window_size) // self.stride + 1
        # x = rearrange(x, 'b (h w) n -> b n h w', h=self.window_size, w=self.window_size)
        x = self.dct1d(x)

        return rearrange(x, 'b c (k1 k2) -> b c k1 k2', k1=self.h, k2=self.w)
    def get_bound(self, x):
        self.dct_matrix = self.dct_matrix.to(x.device)

    def idct_forward(self, x):
        if self.shape_x[-2:] != self.divisor.shape[-2:]:
            h, w = self.shape_x[-2:]
            self.input_ones = torch.ones([1, 1, h, w])
            self.input_ones = self.input_ones.to(x.device)
            self.fold = nn.Fold(output_size=self.shape_x[-2:], **self.fold_params)
            self.divisor = self.fold(self.unfold(self.input_ones))
        if self.divisor.device != x.device:
            self.divisor = self.divisor.to(x.device)
        # x = rearrange(x, 'b c h w -> b c (h w)', h=self.h, w=self.w)
        x = rearrange(x, 'b c k1 k2 -> b c (k1 k2)')
        x = self.idct1d(x)

        x = self.fold(x) / self.divisor
        # print('fold: ', x.shape)
        if self.mode == 'reflect':
            x = x[:, :, self.pad_size:, self.pad_size:].contiguous()
        x = rearrange(x, '(b c) k h w -> b (c k) h w', b=self.b, c=self.c, k=1)
        # out = self.project_out(x)
        return x
    def forward(self, x, dct_forward=True):
        if dct_forward:
            return self.dct_forward(x)
        else:
            return self.idct_forward(x)
        # b, c, H, W = x.shape
        # # _, _, H, W = x.shape
        # x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
        # if self.mode == 'reflect':
        #     x = F.pad(x, (self.pad_size, 0, self.pad_size, 0), mode=self.mode)
        # if x.shape[-2:] != self.divisor.shape[-2:]:
        #     self.input_ones = torch.ones_like(x)
        #     self.fold = nn.Fold(output_size=x.shape[-2:], **self.fold_params)
        #     self.divisor = self.fold(self.unfold(self.input_ones))
        # if self.divisor.device != x.device:
        #     self.divisor = self.divisor.to(x.device)
        # # print('y: ', x.shape)
        # x = self.unfold(x)
        # # print('unfold: ', x.shape)
        #
        # if self.mode != 'reflect':
        #     h, w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
        #                 W + 2 * self.pad_size - self.window_size) // self.stride + 1
        # else:
        #     h, w = (H + self.pad_size - self.window_size) // self.stride + 1, (
        #             W + self.pad_size - self.window_size) // self.stride + 1
        # x = rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
        # x = self.dct2d(x)
        # # x = self.conv2(self.conv1(x))
        # # x = self.act(x)
        # # x = self.conv3(x) + short_cut
        # x = self.idct2d(x)
        # x = rearrange(x, 'b c h w -> b c (h w)', h=h, w=w)
        #
        # x = self.fold(x) / self.divisor
        # # print('fold: ', x.shape)
        # if self.mode == 'reflect':
        #     x = x[:, :, self.pad_size:, self.pad_size:]
        # x = rearrange(x, '(b c) k h w -> b (c k) h w', b=b, c=c, k=1)
        # # out = self.project_out(x)
        # return x
# class DCT2d_fold_branch_torch_dct(nn.Module):
#     def __init__(self, window_size=8, pad_size=0, stride=1, pad_mode='reflect'):
#         super().__init__()
#         self.window_size = window_size
#         # n = window_size ** 2
#         # self.dct2d = DCT2(window_size=window_size)
#         # self.idct2d = IDCT2(window_size=window_size)
#         # self.dct_matrix = get_dctMatrix(window_size, window_size)
#         self.stride = stride
#         self.mode = pad_mode
#         if self.mode != 'reflect':
#             self.pad_size = pad_size
#             self.fold_params = dict(kernel_size=window_size, dilation=1, padding=self.pad_size // 2, stride=self.stride)
#         else:
#             self.pad_size = window_size - 1
#             self.fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=self.stride)
#         # output_size = [1, 1, 128, 128]
#         # self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
#         self.unfold = nn.Unfold(**self.fold_params)
#         #
#         # self.input_ones = torch.ones(output_size)
#         # self.divisor = self.fold(self.unfold(self.input_ones))
#
#     def dct_forward(self, x):
#         self.b, self.c, H, W = x.shape
#         # _, _, H, W = x.shape
#         x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
#         if self.mode == 'reflect':
#             x = F.pad(x, (self.pad_size, 0, self.pad_size, 0), mode=self.mode)
#         # print('y: ', x.shape)
#         x = self.unfold(x)
#         # print('unfold: ', x.shape)
#
#         if self.mode != 'reflect':
#             self.h, self.w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
#                     W + 2 * self.pad_size - self.window_size) // self.stride + 1
#         else:
#             self.h, self.w = (H + self.pad_size - self.window_size) // self.stride + 1, (
#                     W + self.pad_size - self.window_size) // self.stride + 1
#         x = rearrange(x, 'b (h w) n -> b n h w', h=self.window_size, w=self.window_size)
#         x = DCT.dct_2d(x, norm='ortho')
#
#         return rearrange(x, '(b c) (k1 k2) h w -> b (c h w) k1 k2', c=self.c, k1=self.h, k2=self.w)
#     def forward(self, x, dct_forward=True):
#         # if dct_forward:
#         return self.dct_forward(x)
#         # else:
#         #     return self.idct_forward(x)

if __name__=='__main__':
    import torch
    import kornia
    import cv2
    import os

    # net = DCT2d_fold_branch(window_size=8, dct_torch=True).cuda()
    # # net = Window_Local(qkv=False)
    # x = torch.randn(1, 32, 128, 128)
    x = torch.randn(1, 32, 8, 8)
    z = torch.randn(1, 32, 8, 16)
    x = x / 5.
    x = x.cuda()
    # y = dct_2d_torch(x, 'ortho')
    # y_ = idct_2d_torch(x, 'ortho')
    # dct_n = DCT2(8)
    # idct_n = IDCT2(8)
    dct_x = DCT2x()#8,16 
    idct_x = IDCT2x()#8, 16
    z1 = dct_x(z)
    z1 = idct_x(z1)
    print(torch.mean(z1 - z))
    # y2 = dct_n(x)
    # y2_ = idct_n(y2)
    # y2_x = idct_n(x)
    # print(y2.max(), y2.min())
    # print(torch.mean(y2 - y))
    # print(torch.mean(y2_ - x))
    # print(torch.mean(y2_x - y_))
    # y = idct_2d_torch(y)
    # y = idct1_spectral2d_torch(x)
    # print(y.max(), y.min())
    # y = dct1_spectral2d_torch(y)
    # y = net(x, True)
    #
    # y = net(y, False)
    # print(torch.mean(y - x))
    # net = Mestormer(dim=8).cuda()
    # # net = Window_Local(qkv=False)
    # x = torch.randn(1,3,128,128)
    # x = x.cuda()

    # dct1d = DCT1d(window_size=64)
    # idct1d = IDCT1d(window_size=64)
    # Conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, stride=1, padding=1, padding_mode='reflect')
    # Conv2 = Conv2.cuda()
    # window_size = 8
    # pad_size = 7
    # mode = 'reflect' # 'contast' #
    # stride = 1
    # img_size = 256
    # if mode != 'reflect':
    #     fold_params = dict(kernel_size=window_size, dilation=1, padding=pad_size, stride=stride)
    #     fold_params_fold = dict(kernel_size=window_size, dilation=1, padding=pad_size, stride=stride)
    #     # fold_params_fold['stride'] = 2 * stride
    #     fold = nn.Fold(output_size=[img_size//2, img_size//2], **fold_params_fold)
    # else:
    #     fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=stride)
    #     fold_params_fold = dict(kernel_size=window_size, dilation=1, padding=0, stride=stride)
    #     output_size = (img_size+pad_size) # // 2
    #     fold = nn.Fold(output_size=[output_size, output_size], **fold_params_fold)
    #
    # unfold = nn.Unfold(**fold_params)
    # # z = torch.randn(3, 1, 128, 128)
    # z = cv2.imread('/home/mxt/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/val/target_crops/0.png')
    # z = kornia.image_to_tensor(z, keepdim=False)
    # z_ = kornia.color.rgb_to_grayscale(z/255.)
    # z = z_.cuda()
    # H, W = z.shape[-2:]
    # if mode == 'reflect':
    #     z = F.pad(z, (pad_size, 0, pad_size, 0), 'reflect')
    #
    # input_ones = torch.ones_like(z)
    # divisor = fold(unfold(input_ones))
    # y = unfold(z)
    # if mode != 'reflect':
    #     h, w = (H + 2 * pad_size - window_size) // stride + 1, (
    #                 W + 2 * pad_size - window_size) // stride + 1
    # else:
    #     h, w = (H + pad_size - window_size) // stride + 1, (
    #             W + pad_size - window_size) // stride + 1
    #
    # y = dct1d(y)
    # y = rearrange(y, 'b c (h w) -> b c h w', h=h, w=w)
    # y = Conv2(y)
    # y = torch.relu(y)
    # print(y.shape)
    # y_out = kornia.tensor_to_image(y.cpu())
    # out_dir = '/home/mxt/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/val/freq/dct_fold_sharp_ref_s2'
    # os.makedirs(out_dir, exist_ok=True)
    # for i in range(y_out.shape[-1]):
    #     cv2.imwrite(os.path.join(out_dir, str(i)+'.jpg'), y_out[:,:,i]*255)
    # y = rearrange(y, 'b c h w -> b c (h w)', h=h, w=w)
    # y = idct1d(y)
    # y = fold(y) / divisor
    # print(y.shape)
    # print(torch.sum(y-z))
    # if mode == 'reflect':
    #     y = y[:, :, pad_size:, pad_size:]
    # print(torch.sum(y - z_.cuda()))
    # y = kornia.tensor_to_image(y.cpu())
    # cv2.imwrite(os.path.join(out_dir, 'y.jpg'), y * 255)
    # y = torch.randn(8, 32, 8, 8)
    # y1 = rearrange(y, 'b (head c) (k1 h) (k2 w) -> b head (h w) c (k1 k2)', head=4, k1=8, k2=8)
    # y1 = torch.nn.functional.normalize(y1, dim=-2)
    # y1 = rearrange(y1, 'b head (h w) c (k1 k2) -> b head c (k1 h k2 w)', head=4, h=1, w=1, k1=8, k2=8)
    # y2 = rearrange(y, 'b (head c) h w -> b head c (h w)', head=4)
    # y2 = torch.nn.functional.normalize(y2, dim=-2)
    # print(torch.mean(y1-y2))
    # y, batch_list = window_partitionxy(x, 32, start=[16, 16])
    # out = window_reversexy(y, 32, 128, 128, batch_list, start=[16, 16])
    # print(torch.mean(out-x))
    # a = torch.fft.hfft2(x)
    # print(a.shape)
    # z = torch.fft.ihfft2(a)
    # print(z.shape)
    # print(torch.mean(z.real-x))
