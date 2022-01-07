# DeepMIH: Deep Invertible Network for Multiple Image Hiding

This repo is the official code for

* [*DeepMIH: Deep Invertible Network for Multiple Image Hiding*](https:) 
  * [*Zhenyu Guan*](http://cst.buaa.edu.cn/info/1071/2542.htm)<sup>1</sup>, [*Junpeng Jing*](https://tomtomtommi.github.io/)<sup>1</sup>(**co-first**), [*Xin Deng*](http://www.commsp.ee.ic.ac.uk/~xindeng/), [*Mai Xu*](http://shi.buaa.edu.cn/MaiXu/zh_CN/index.htm), *Lai Jiang*, *Zhou Zhang*, *Yipeng Li*.

Published on **IEEE Transactions of Pattern Analysis and Machine Intelligence (TPAMI 2022)**.
@ [Beihang University](http://ev.buaa.edu.cn/).

<center>
  <img src=https://github.com/TomTomTommi/DeepMIH/blob/main/image/figure2.jpg width=50% />
</center>


## 1. Pre-request
### 1.1 Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [PyTorch = 1.0.1](https://pytorch.org/) .
- See [environment.yml](https://github.com/TomTomTommi/HiNet/blob/main/environment.yml) for other dependencies.

### 1.2 Dataset

- In this paper, we use the commonly used dataset DIV2K, COCO, and ImageNet.
- For train or test on your own path, change the code in `config.py`:
    `line50:  TRAIN_PATH_DIV2K = '' ` 
    `line51:  VAL_PATH_DIV2K = '' `
    `line54:  VAL_PATH_COCO  = '' `
    `line55:  TEST_PATH_COCO = '' ` 
    `line57:  VAL_PATH_IMAGENET  = '' `

## 2. Test

1. Here we provide a trained [model](https://drive.google.com/drive/folders/1guno6VwfCpuB8o5m0ZqFHNL4ZWc8SdJe?usp=sharing).
2. Download and update the `MODEL_PATH` and the file name `suffix` before testing by the trained model.
For example, if the model name is `model_1.pt`, `model_2.pt`, `model_3.pt` and its path is `/home/usrname/Hinet/model/`, 
set `PRETRAIN_PATH = '/home/usrname/Hinet/model/'`, `PRETRAIN_PATH_3 = '/home/usrname/Hinet/model/'` and file name `suffix = 'model.pt'`.
3. Check the dataset path is correct.
4. Create an image path to save the generated images. Update the `TEST_PATH`.
5. Run `test_oldversion.py`.


## 3. Train

1.
4. Run `train_old_version.py`.


## Citation
If you find this repository helpful, you may cite:

```tex
@article{
}
```
