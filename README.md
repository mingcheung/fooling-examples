## About
PyTorch code for our paper: **Specious Examples: Another Intriguing Property of Neural Networks**.
<br> Ming Zhang, Yongkang Chen, Cheng Qian.


### Requirements
torch>=1.7.0; torchvision>=0.8.1; tqdm>=4.31.1; pillow>=7.0.0; matplotlib>=3.2.2;  numpy>=1.18.1; 

### Dataset
The 1000 images from the NIPS 2017 ImageNet-Compatible dataset are provided in the folder ```dataset/images```, along with their metadata in  ```dataset/images.csv``` and ```dataset/imagenet_class_index.json```.

### Evaluation
```gen_initial_images.py```: Generate initial images, including random Gaussian noised, random uniform noised, all-white and all-black images. 

```eval_single.py```: Generate specious examples on a single model.

```eval_ensemble.py```: Generate specious examples on an ensemble of models. 

```cal_succ_rate```: Calculate the accuracy of specious or normal examples. 

```plot_figures.py```: Plot figures in experiments.

```utils.py```: Some necessary utility functions.
