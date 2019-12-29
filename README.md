# Binocular Deep Neural Network for DR Detection Combined with Area Extraction

## Introductions to the four folders 

- The first folder named efficientnet_pytorch contain the core implement model from efficientnet-b0 to efficientnet-b7
- The second folder named examples contains the core implement of our Binocular Deep Neural Network for DR Detection Combined with Area Extraction. Our model is built based the implement of efficientnet in the first folder
- The third folder named tests is used to test implement of our efficientnet
- The last folder named tf_pytorch is used to load pretrained models for efficientnet

## Detail Introductions to the Second Folder, Namely Our Binocular Deep Neural Network Model

- **examples/data_processing:** It contains codes for data augment and processing

- **examples/simple/model/dataset.py:** Custom dataset to load the images locally

- **examples/simple/model/resnet.py:** A rather simple implement of  a quite simple resnet model. We use it to test the performance of deep neural network for DR detection at the beginning of our project

- **examples/simple/model/test_model.py:** In this file we manually load the model we have trained and test performance of our model and baselines

- **examples/simple/model/trainer.py:** This file contains the core implement of the training procedure of our model and baselines

- **examples/simple/model/our_model.py:** This file contains the core implement of our Binocular Deep Neural Network model including the model for area extraction and classification

- **examples/simple/model/universal_net.py:** Entrance of our project. You can train different models via different commands (We will introduce later). Models besides our binocular model include 

  ```python
  "resnet18": torchvision.models.resnet18(pretrained=args.pretrain),
  "resnet34": torchvision.models.resnet34(pretrained=args.pretrain),
  "resnet50": torchvision.models.resnet50(pretrained=args.pretrain),
  "resnet101": torchvision.models.resnet101(pretrained=args.pretrain),
  "resnet152": torchvision.models.resnet152(pretrained=args.pretrain),
  "densenet121": torchvision.models.densenet121(pretrained=args.pretrain),
  "densenet": torchvision.models.densenet161(pretrained=args.pretrain),
  "densenet169": torchvision.models.densenet169(pretrained=args.pretrain),
  "densenet201": torchvision.models.densenet201(pretrained=args.pretrain),
  "wide_resnet50_2": torchvision.models.wide_resnet50_2(pretrained=args.pretrain),
  "wide_resnet101_2": torchvision.models.wide_resnet101_2(pretrained=args.pretrain)
  "resnext101_32*8d":torchvision.models.resnext101_32x8d(pretrained=args.pretrain)
  ```

## Guide on usage of universal_net.py

- Firstly modify the path of your data( it is named "path_list" in the code file) and the saved model( it is named "save_path" in the code file)
- Secondly load the trained model for area extraction (you need to load it in this file manually, hints are included in this file)

- Run a specific model (models you can use are listed above)

  ```shell
  python universal_net.py --model_detail efficientnet-b3 --sort_kinds 2 --image_size 224 --batch_size 64
  ```

- If the model is too large, you can 

  - change the image_size argument or change the batch_size

  - run another file and change the argument (this model is rather small)

    ```shell
    python resnet.py --image_size 30 --batch_size 64
    ```

- you just need to change the image_size and batch_size to find better hyper-parameters

## Guide on test_model.py

- Firstly modify the path of your data( it is named "path_list" in the code file) and the saved model( it is named "save_path" in the code file)

- Secondly run test_model.py. Only three arguments are needed

  ```python
  python test_model.py --sort_kinds 2(/4/5) --model_detail resnet18 --image_size 224 
  ```
