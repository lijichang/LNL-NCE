# LNL-NCE
A pytorch implementation for "Neighborhood Collective Estimation for Noisy Label Identification and Correction", accepted by ECCV2022. More details of this work can be found in our paper: [Arxiv](https://arxiv.org/abs/2208.03207) or [ECCV2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840126.pdf).


## Installation

Refer to  [DivideMix](https://github.com/LiJunnan1992/DivideMix).

## Model training

(1) To run training on CIFAR-10/CIFAR-100 with different noise modes (namely **sym** or **asym**) and various noise ratios (namely **0.20**, **0.50**, **0.80**, **0.90**, etc.),

`CUDA_VISIBLE_DEVICES=0 python ./cifar/main.py --dataset cifar10 --num_class 10 --batch_size 128 --data_path ./data/cifar-10/ --r 0.50 --noise_mode sym --remark exp-ID`

`CUDA_VISIBLE_DEVICES=0 python ./cifar/main.py --dataset cifar100 --num_class 100 --batch_size 128 --data_path ./data/cifar-100/ --r 0.50 --noise_mode sym --remark exp-ID`

(2) To run training on Webvision-1.0,
`CUDA_VISIBLE_DEVICES=0,1,2 python ./webvision/main.py --data_path ./data/webvision/ --remark exp-ID`

### Citation
If you consider using this code or its derivatives, please consider citing:

```
@inproceedings{li2022neighborhood,
  title={Neighborhood Collective Estimation for Noisy Label Identification and Correction},
  author={Li, Jichang and Li, Guanbin and Liu, Feng and Yu, Yizhou},
  booktitle={European Conference on Computer Vision},
  pages={128--145},
  year={2022},
  organization={Springer}
}
```
### Contact
Please feel free to contact the first author, namely [Li Jichang](https://lijichang.github.io/), if you have any questions.
