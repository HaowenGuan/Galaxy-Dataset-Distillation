import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')
#%%

dir = '/Users/data/class9/demonstrate_of_each_class'
output = np.zeros((512, 512 * 10, 3))
for i in range(10):
    img = cv2.imread(os.path.join(dir, str(i) + '.png'))
    output[:, i * 512: (i + 1) * 512] = img
cv2.imwrite('/Users/data/class9/gzoo2-1-per-class-demonstration-0-9.jpg', output)



#%%
print('wtf')
#%%
from PIL import Image
pil_image = Image.open("/Users/data/class9/demonstrate_of_each_class/5.png")
# im.rotate(18).show()
pil_image = np.array(pil_image)[:, :, :3]

#%%
import cv2
cv_image = cv2.imread("/Users/data/class9/demonstrate_of_each_class/5.png")
# image = cv2.rotate(src, 45)
# print(image)
cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
#%%
import numpy as np
s = list(range(10))
np.random.shuffle(s)
print(s)
#%%
a = 0
prime = [2, 3, 5, 7, 11, 13, 17, 19]
for i in prime:
    a += 1 << i
print(a, bin(a), len(bin(a)))
#%%
import torch
log_syn_lr = torch.tensor(10.0).requires_grad_(True)
optimizer_lr = torch.optim.SGD([log_syn_lr], lr=lr_lr, momentum=0)
temp = torch.tensor(0.001) * log_syn_lr.data * log_syn_lr
temp.backward()
optimizer_lr.step()
optimizer_lr.zero_grad()
#%%
x = np.array(range(200))
y = np.random.normal(size=200)
corr = np.correlate(x,y)
#%%
import torch
x = torch.randn(2, 3, 10, 10)
orig_shape = x.size()
#%%
# Reshape and calculate positions of top 10%
x = x.view(x.size(0), x.size(1), -1)
#%%
nb_pixels = x.size(2)
ret = torch.topk(x, k=int(0.1*nb_pixels), dim=2)
ret.indices.shape
#%%
# Scatter to zero'd tensor
res = torch.zeros_like(x)
res.scatter_(2, ret.indices, ret.values)
res = res.view(*orig_shape)
#%%

#%%

#%%

#%%
