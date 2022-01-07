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
For example, if the model name is  `model_checkpoint_03000_1.pt`,  `model_checkpoint_03000_2.pt`,  `model_checkpoint_03000_3.pt`,  
and its path is `/home/usrname/DeepMIH/model/`,  
set:  
`PRETRAIN_PATH = '/home/usrname/DeepMIH/model/'`,  
`PRETRAIN_PATH_3 = '/home/usrname/DeepMIH/model/'`,  
file name `suffix = 'model_checkpoint_03000'`.  
3. Check the dataset path is correct.
4. Create an image path to save the generated images. Update `TEST_PATH`.
5. Run `test_oldversion.py`.


## 3. Train

1. Run `train_old_version.py`. Following the Algorithm 1 to train the model.
2. **Note: MFQEv2 may be hard to train.** The model may suffer from explosion. Our solution is to stop the training process at a normal node and abate the learning rate. Then, continue to train the model.


## 4. Further explanation
In the `train_old_version.py` at line 223:  
`rev_secret_dwt_2 = rev_dwt_2.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12`,   
the recovered secret image_2 is obtained by spliting the middle 12 channels of the varible `rev_dwt_2`. However, in the forward process_2, the input is obtained by concatenating (stego, imp, secret_2) together. This means that the original code `train_old_version.py` has a bug on recovery process (the last 12 channels of the varible `rev_dwt_2` should be splited to be the recovered secret image_2, instead of the middle 12 one). We found that in this way the network is still able to converge, thus we keep this setting in the test process.  
We also offer a corrected version `train.py` (see line 225) and `test.py`. You can also train your own model in this way.


## Citation
If you find this repository helpful, you may cite:

```tex
@article{
}
```
