import cv2
import numpy as np
import os

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
im = Image.open("/Users/data/class9/demonstrate_of_each_class/5.png")
im.rotate(18).show()

#%%
import cv2
src = cv2.imread("/Users/data/class9/demonstrate_of_each_class/5.png")
image = cv2.rotate(src, 45)
print(image)
cv2.imshow('Image', image)
#%%

#%%
