import torch.nn as nn
import torch
from torchvision import models

class AdaptiveConcatPool2d(nn.Module):
    """
    Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`.
    Source: Fastai. This code was taken from the fastai library at url
    https://github.com/fastai/fastai/blob/master/fastai/layers.py#L176
    """
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    
class Flatten(nn.Module):
    """
    Flatten `x` to a single dimension. Adapted from fastai's Flatten() layer,
    at https://github.com/fastai/fastai/blob/master/fastai/layers.py#L25
    """
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)

def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn=None):
    """
    Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`.
    Adapted from Fastai at https://github.com/fastai/fastai/blob/master/fastai/layers.py#L44
    """
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers
    
def create_head(top_n_tags, nf, ps=0.5):
    nc = top_n_tags
    
    lin_ftrs = [nf, 512, nc]
    p1 = 0.25 # dropout for second last layer
    p2 = 0.5 # dropout for last layer

    actns = [nn.ReLU(inplace=True),] + [None]
    pool = AdaptiveConcatPool2d()
    layers = [pool, Flatten()]
    
    layers += [
        *bn_drop_lin(lin_ftrs[0], lin_ftrs[1], True, p1, nn.ReLU(inplace=True)),
        *bn_drop_lin(lin_ftrs[1], lin_ftrs[2], True, p2)
    ]
    
    return nn.Sequential(*layers)

def _resnet(base_arch, top_n, **kwargs):
    cut = -2
    s = base_arch(pretrained=False, **kwargs)
    body = nn.Sequential(*list(s.children())[:cut])

    if base_arch in [models.resnet18, models.resnet34]:
        num_features_model = 512
    elif base_arch in [models.resnet50, models.resnet101]:
        num_features_model = 2048

    nf = num_features_model * 2
    nc = top_n

    head = create_head(nc, nf)
    model = nn.Sequential(body, head)

    return model

def resnet18(pretrained=True, progress=True, top_n=100, **kwargs):
    r""" 
    Resnet18 model trained on the Danbooru2018 dataset tags

    Args:
        pretrained (bool): kwargs, load pretrained weights into the model
        top_n (int): kwargs, pick to load the model for predicting the top `n` tags. 
            currently only supports top_n=100.
    """
    
    model = _resnet(models.resnet18, top_n, **kwargs)
    if pretrained:
        if top_n == 100: 
            state = torch.hub.load_state_dict_from_url("https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet18-3f77756f.pth", 
                                                   progress=progress)
            # state = torch.load('weights/resnet18.pth')
            model.load_state_dict(state)
        else:
            raise ValueError("Sorry, the resnet18 model only supports the top-100 tags \
                at the moment")
    
    return model

def resnet34(pretrained=True, progress=True, top_n=500, **kwargs):
    r""" 
    Resnet34 model trained on the Danbooru2018 dataset tags

    Args:
        pretrained (bool): kwargs, load pretrained weights into the model
        top_n (int): kwargs, pick to load the model for predicting the top `n` tags,
            currently only supports top_n=500.
    """
    
    model = _resnet(models.resnet34, top_n, **kwargs)
    if pretrained:
        if top_n == 500: 
            state = torch.hub.load_state_dict_from_url("https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet34-88a5e79d.pth", 
                                                   progress=progress)
            # state = torch.load('weights/resnet34.pth')
            model.load_state_dict(state)
        else:
            raise ValueError("Sorry, the resnet34 model only supports the top-500 tags \
                at the moment")
    
    return model

def resnet50(pretrained=True, progress=True, top_n=6000, **kwargs):
    r""" 
    Resnet50 model trained on the full Danbooru2018 dataset's top 6000 tags

    Args:
        pretrained (bool): kwargs, load pretrained weights into the model.
        top_n (int): kwargs, pick to load the model for predicting the top `n` tags,
            currently only supports top_n=6000.
    """
    model = _resnet(models.resnet50, top_n, **kwargs)
    if pretrained:
        if top_n == 6000: 
            state = torch.hub.load_state_dict_from_url("https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet50-13306192.pth", 
                                                   progress=progress)
            # state = torch.load('weights/resnet50.pth')
            model.load_state_dict(state)
        else:
            raise ValueError("Sorry, the resnet50 model only supports the top-6000 tags \
                at the moment")

    
    return model