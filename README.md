# Pytorch pretrained resnet models for Danbooru2018
This repository contains config info and notebook scripts used to train several ResNet models for predicting the tags of images in the Danbooru2018 dataset. An example of the resnet50's output is shown below.

![img1](img/danbooru_resnet2.png)

For a rundown of using these networks, training them, the performance of each network, and other useful information, please [see the accompanying post on this](https://rf5.github.io/2019/07/08/danbuuro-pretrained.html). As in the post, it only takes a few lines of pure pytorch to get started. 

__Requirements__:
- Pytorch (>1.0)
- (optional) fastai

## Getting Started
Lets predict the tags for an image using the resnet50 model. Note that you do not need to clone or download this repository, it is linked to pytorch hub and the following code will work as long as you have pytorch :).
```python
import torch
# Load the model
model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
model.eval()
```

```python
from PIL import Image
import torch
from torchvision import transforms
input_image = Image.open("img/danbooru_resnet1.png") # load an image of your choice
preprocess = transforms.Compose([
    transforms.Resize(360),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# The output has unnormalized scores. To get probabilities, you can run a sigmoid on it.
probs = torch.sigmoid(output[0]) # Tensor of shape 6000, with confidence scores over Danbooru's top 6000 tags
```
Thats it! To plot the results next to the image, you can just run:
```python
import matplotlib.pyplot as plt
import json
import urllib, urllib.request
# Get class names
with urllib.request.urlopen("https://github.com/RF5/danbooru-pretrained/raw/master/config/class_names_6000.json") as url:
    class_names = json.loads(url.read().decode())
# Plot image
plt.imshow(input_image)
plt.grid(False)
plt.axis('off')

def plot_text(thresh=0.2):
    tmp = probs[probs > thresh]
    inds = probs.argsort(descending=True)
    txt = 'Predictions with probabilities above ' + str(thresh) + ':\n'
    for i in inds[0:len(tmp)]:
        txt += class_names[i] + ': {:.4f} \n'.format(probs[i].cpu().numpy())
    plt.text(input_image.size[0]*1.05, input_image.size[1]*0.85, txt)

plot_text()
plt.tight_layout()
plt.show()
```
![img2](img/danbooru_resnet1.png)

For more info on the performance and other ways to load the network, see [this link](https://rf5.github.io/2019/07/08/danbuuro-pretrained.html).

## Files
- `config/` contains the class names for the various number of top tags that the network predicts. For example, the resnet50's 5th output is the (unnormalized) probability of the image containing the 5th tag name in `class_names_6000.csv`.
- `training_notebooks/` contains notebooks which I based my training of the networks on. The resnet34 notebook is entirely similar to the resnet18 notebook.
- `danbooru_resnet.py` contains functions to build and load the various danbooru resnet networks (again see the [blog post](https://rf5.github.io/2019/07/08/danbuuro-pretrained.html) for details)

## References
1. Thanks a ton for the organizers of the [Danbooru2018 dataset](https://www.gwern.net/Danbooru2018)! Their citation is: 
> Anonymous, The Danbooru Community, Gwern Branwen, & Aaron Gokaslan; “Danbooru2018: A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset”, 3 January 2019. Web. Accessed 2019-06-24. 
2. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
3. [Fastai](https://docs.fast.ai)

## Citing
If you use the pretrained models and feel it was of value, please do give some form of shout-out, or if you prefer you may use the bibtex entry:
```tex
@misc{danbooru2018resnet,
    author = {Matthew Baas},
    title = {Danbooru2018 pretrained resnet models for PyTorch},
    howpublished = {\url{https://rf5.github.io}},
    url = {https://rf5.github.io/2019/07/08/danbuuro-pretrained.html},
    type = {pretrained model},
    year = {2019},
    month = {July},
    timestamp = {2019-07-08},
    note = {Accessed: DATE}
}

