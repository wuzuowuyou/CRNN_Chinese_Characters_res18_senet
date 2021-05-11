# Characters Recognition

A Chinese characters recognition repository based on convolutional recurrent networks. 

from https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec


## Dev Environments
1. WIN 10 or Ubuntu 16.04
2. **PyTorch 1.7.0 (may fix ctc loss)** with cuda 10.2 🔥（其他版本的应该也可以）
3. yaml
4. easydict

## 相比原版CRNN_Chinese_Characters_Rec做出的改动
1. 增加在线数据增强操作
2. 增加resnet18和resnet18+senet基网络
3. 图片可以任意大小，在数据预处理部分可以使得图片不变形归一化到固定大小，具体可以看代码
4. 修改数据读取方式
5. 所有的配置可在./lib/config/OWN_config.yaml文件修改
6. 下载本git，配置好环境，可直接python train.py训练。已经放置了数据可以直接训练。
7. 小量数据只是为了能跑通，并不是看loss，acc的，需要大几十万的数据才能训练出高的准确度。

## 数据准备
1. 按照如下格式准备
```
.
├── train
│   ├── 1
│   │   ├── 27762750_1861112355_CDMAUE的无线通.jpg
│   │   ├── 29575671_1837255940_在这样短的航行时间里.jpg
│   │   └── 30471375_367454767_题为“满足哮喘患者的.jpg
│   ├── 2
│   │   ├── 28151468_416830229_社会学、管理学、法学.jpg
│   │   └── 30308937_625908317_日军在打开石门缺口之.jpg
│   ├── 29862718_3433251563_南京大屠杀.jpg
│   └── 3
│       ├── 28946890_3487470386_现在还没有明白这是怎.jpg
│       ├── 29363812_2810713842_什么用什么该自己做主.jpg
│       └── 30051984_3546758428_且裁判也将手指向了中.jpg
└── val
    ├── 1
    │   └── 29575671_1837255940_在这样短的航行时间里.jpg
    ├── 2
    │   └── 30308937_625908317_日军在打开石门缺口之.jpg
    ├── 28971703_2070257603_鲜花.jpg
    └── 3
        └── 28946890_3487470386_现在还没有明白这是怎.jpg

```
2. **其中文件夹train和val，两个文件夹里面随便放多少个目录和随便放图片，但是有个原则就是图片命名最后的"_"和最后的"."之间是label**


## 3. 训练与测试
1. 数据按照规定存放，配置/lib/config/OWN_config.yaml
2. python train.py
3. python demo_show.py


## References
- https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec
- https://github.com/meijieru/crnn.pytorch
- https://github.com/HRNet

If this repository helps you，please star it. Thanks.




