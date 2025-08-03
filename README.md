# Feature2Pixel

一个用于图像特征到图像像素转换的深度学习项目。

## 🔧 环境配置

建议使用 [Anaconda](https://www.anaconda.com/) 来管理环境：

```bash
conda create -n F2P python=3.9
conda activate F2P

克隆本仓库并安装依赖：

```bash
git clone https://github.com/yourusername/Feature2Pixel-main.git
cd Feature2Pixel-main
pip install -r requirements.txt
```

## 🚀 快速开始

确保你已经准备好了数据集（如 Kodak24），然后运行以下命令开始训练：

```bash
mkdir log  # 若没有 log 文件夹，需手动创建
python Feature2Pixel_syn.py
```

你可以根据自己的需求在 `Feature2Pixel_syn.py` 中修改数据路径和参数设置。

## 📁 项目结构

```
Feature2Pixel-main/
├── Feature2Pixel_syn.py         # 主程序（合成噪声实验）
├── models/                      # 模型模块（特征提取与图像恢复网络）
├── utils.py                     # 工具函数（如Logger、图像可视化等）
├── options/                     # 参数设置模块
├── datasets/                    # 数据集预处理与加载模块
├── requirements.txt             # 所需依赖库
└── README.md                    # 本文件
```

## 🧪 数据集说明

你可以使用以下数据集进行训练与评估：

* **Kodak24**
* **BSD68**
* **Urban100**
* 或你自己的图像数据集

示例路径：`/yourpath/dataset/Kodak24_c256_noisy/gauss_nl50/kodim04.png`

## 🖼️ 输出与评估

* 图像保存路径可在脚本中设置
* 默认输出图像、日志文件（保存在 `log/` 文件夹）
* 支持 PSNR / SSIM 评估指标（需要你集成或添加代码）

## 🔗 项目主页与代码

项目主页：[GitHub - Feature2Pixel](https://github.com/Tangfangcai/Feature2Pixel)


## 📧 联系方式

如有问题，请提交 Issue






本项目采用 [CC BY-NC-SA 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh)。  
This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
