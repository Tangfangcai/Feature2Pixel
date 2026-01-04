# Feature2Pixel

一个零样本去噪方法。

## 📝 TODO
- [ ] 上传完整PixelFeatureMap.py代码

## 🔧 环境配置

建议使用[Anaconda]或者[miniconda] (https://www.anaconda.com) 来管理环境：

```bash
conda create -n F2P python=3.9
conda activate F2P
```

克隆本仓库并安装依赖：

```bash
git clone https://github.com/tangfangcai/Feature2Pixel.git
cd Feature2Pixel
pip install -r requirements.txt
```
✅ 安装 PyTorch（必选）
请根据你电脑的 CUDA 版本或是否使用 GPU，访问 https://pytorch.org/ ，选择适合你的配置并复制相应的安装命令。
例如，如果你使用的是 Linux + CUDA 11.8 + pip，可以执行：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 快速开始

运行以下命令开始：

```bash
python Feature2Pixel_real.py
```

你可以根据自己的需求在 `Feature2Pixel_syn.py` 中修改数据路径和参数设置。


## 🧪 数据集说明

你可以使用以下数据集进行训练与评估：在data 文件夹中创建你的数据集文件夹。数据集文件夹应包含两个子文件夹：GT 和 Noisy，分别存储干净图像和噪声图像。

```
Feature2Pixel-main/
└──  data 
    └──  your_dataset_name
        ├──  GT
            ├── pic1.png
            └── pic2.png
        └──  Noisy
            ├── pic1.png
            └── pic2.png
```


## 🖼️ 输出与评估

* 图像保存路径可在脚本中设置
* 默认输出图像、日志文件（保存在 `log/` 文件夹）
* 支持 PSNR / SSIM 评估指标

## 🔗 项目主页与代码

项目主页：[GitHub - Feature2Pixel](https://github.com/Tangfangcai/Feature2Pixel)


## 📧 联系方式

如有问题，请提交 Issue


本项目采用 [CC BY-NC-SA 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh)。  
This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
