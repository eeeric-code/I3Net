import torch
from torch.autograd import Variable
import numpy as np


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]

def normalize(slice):
    ma,mi=4095 , 0
    slice = (slice - mi)/(ma - mi)
    
    return slice

class RandomCrop3d(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples)
        return transform(samples)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample.astype(np.float32))

def prepare(args,gpu_id=0,precision='half'):
    device = torch.device(f'cuda:{gpu_id}')
    def _prepare(tensor):
        if precision == 'half': tensor = tensor.half()
        return tensor.to(device)
        
    # return [_prepare(a) for a in args]
    return _prepare(args)

