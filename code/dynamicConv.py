import torch
import torch.nn as nn
import torch.nn.functional as F

def print_size(w):
    print(w.size(0), w.size(1), w.size(2), w.size(3))

class Attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(Attention2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if in_planes == 3:
            hidden_planes = K
        else:
            hidden_planes = int(in_planes * ratios) + 1
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, 1, bias=False)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def update_temperature(self, epoch):
        if epoch < 10:
            self.temperature -= 3
            # print('temperature =', self.temperature)
        elif epoch == 10:
            self.temperature = 1
            # print('temperature =', self.temperature)
        else:
            pass
            # print('keep temperature =', self.temperature)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = F.relu(x)
        # print('in attention2d: x size0 =', x.size(0))
        x = self.fc2(x).view(x.size(0), -1)
        # print('in attention2d: x size1 =', x.size)
        x = F.softmax(x / self.temperature, 1)
        # print('in attention2d: x final =', x)
        return x


class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, K=4, temperature=1, init_weight=True):
        super(DynamicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.K = K
        self.attention = Attention2d(in_channels, 0.25, K, temperature, init_weight)

        # weight shape are 
        self.weight = nn.Parameter(torch.randn(K, out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self, epoch):
        self.attention.update_temperature(epoch)

    def forward(self, x):
        softmax_atten = self.attention(x)
        # print('soft atten')
        # print(softmax_atten.size(0), softmax_atten.size(1))
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width) # 1*K 
        # print('original weight')
        # print_size(self.weight)
        weight = self.weight.view(self.K, -1)
        # print('reshape weight')
        # print(weight.size(0), weight.size(1))

        aggregate_weight = torch.mm(softmax_atten, weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        # print('agregate weight')
        # print_size(aggregate_weight)
        if self.bias:
            aggregate_bias = torch.mm(softmax_atten, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, groups=batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, groups=batch_size)
        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))

        return output


