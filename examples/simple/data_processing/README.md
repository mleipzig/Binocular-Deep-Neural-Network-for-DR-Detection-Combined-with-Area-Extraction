# Data augmentation

Function:

datagen(x,y) : x and y are ndarray in which x is the input image and y is the corresponding label

randomly rotate and flip

# CLAHE

Function:

CLAHE(img): input an img and return an img

# Crop and Gaussian Blur

Function:

crop_and_gaussian()

# Note

I have generated about 2000 images per category and treat them with crop_and_gaussian and CLAHE respectively. 

Each 2000 images take about 10GB space in the form of .npy , which means there are 3 kinds of training dataset and each takes about 40-50 GB.

If you think transferring the data takes plenty of time, maybe you can generate the images yourself. However, from my perspective, that may even cause more. Anyway, when I utilize scp, the speed is about 10MB/s and that seems OK for me.

# About loading data

To avoid loading large amount of data into the memory once, you can use the following code:

```python
import numpy as np
x=np.load(path,mmap_mode='r')

```

