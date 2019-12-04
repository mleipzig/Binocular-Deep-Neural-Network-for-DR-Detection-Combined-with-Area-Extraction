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

# Usage of Resnet Model

- install pytorch-gpu and tensorboardX

- clone the  repo, add set PYTHONPATH and get into the work directory

```shell
git clone https://github.com/xiangyu-liu/EfficientNet-PyTorch.git
cd EfficientNet-PyTorch
export PYTHONPATH=./:$PYTHONPATH
cd ./example/simple
```

- modify the path where data is stored. It is in the main.py file. Change it to you data path

```python
path_list = ["/newNAS/Workspaces/DRLGroup/xiangyuliu/NEW/x_0.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/NEW/x_1.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/NEW/x_2.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/NEW/x_3.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/NEW/x_4.npy"]
```

- run a specific model

```shell
python universal_net.py --model_detail resnet18 --image_size 30 --batch_size 64
```

- if the model is too large, you can:

  - change the image_size argument or change the batch_size

  - run another file and change the argument.

    ```shell
    python resnet.py --image_size 30 --batch_size 64
    ```

- you just need to change the image_size and batch_size to find the better hyper-parameters

# Guide on test_model.py

- first modify the of your data( it is name path_list in the code file) and the saved model( it is named save_path in the code file)

- second run test_model.py. Only three arguments are needed

  ```python
  python test_model.py --model_type four(/binary) --model_detail resnet18 --image_size 224 
  ```

  If you model_type is binary the model_detail is not needed

# Guide on New Model

You only need to change two arguments to train different models 

- --model_detail (resnet18 or effcientnet-b0(0-7), etc)
- --sort_kinds(2 or 4 or 5)

eg:

```shell
pthon universal_net --model_detail efficientnet-b3 --sort_kinds 2 --image_size 224 --batch_size 64
```

