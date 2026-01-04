import torch
import torch.nn as nn
import torch.nn.init as init

# -------------------------------
class DenoiseNet(nn.Module):
    def __init__(self, n_chan, chan_embed=64, is_syn=False):
        super(DenoiseNet, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.use_sigmoid = is_syn

        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv4 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv6 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)

        # self.conv7 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        # self.conv8 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        # self.conv9 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)

        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        # x = self.act(self.conv7(x))
        # x = self.act(self.conv8(x))
        # x = self.act(self.conv9(x))

        x = self.conv3(x)
        if self.use_sigmoid:
            return torch.sigmoid(x)
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


# -------------------------------
if __name__ == '__main__':
    # 设置网络参数
    n_chan = 3  # 输入通道数
    chan_embed = 64  # 嵌入通道数
    is_syn = True  # 是否使用 Sigmoid 激活函数

    # 创建网络实例
    net = DenoiseNet(n_chan, chan_embed, is_syn)
    print("网络结构：")
    print(net)

    # 创建一个随机输入张量
    input_tensor = torch.randn(1, n_chan, 256, 256)  # 示例输入，1个样本，3个通道，256x256大小
    output_tensor = net(input_tensor)
    print("输入张量形状：", input_tensor.shape)
    print("输出张量形状：", output_tensor.shape)

    # 计算网络参数数量
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("总参数数量：", total_params)
    print("可训练参数数量：", trainable_params)