import timm
import torch
from pprint import pprint
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


loader = transforms.Compose([transforms.ToTensor()])
image_ = "2.jpg"
unloader = transforms.ToPILImage()


def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)


ttt = image_loader(image_)
print(ttt.size())


random_seed = 78
torch.manual_seed(random_seed)

m = timm.create_model('efficientnet_b3', pretrained=True)
m.eval()

o = m(ttt)
print(o.size())

with torch.no_grad():
    predict = torch.softmax(o, dim=1)
    print(predict.size())
    predict_cal = torch.argmax(predict, dim=1).numpy()
    x1x = predict.numpy()
    x2x = np.argmax(x1x)
# print(o)

print(predict_cal)
print(x2x)
print(x1x[0, x2x])

