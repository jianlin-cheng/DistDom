import torch
import torch.nn as nn
import torch.nn.functional as F
from UNET import UNet
from UNET_1D import UNet_1d
from Attention import Attention
from torch.nn import MultiheadAttention
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.resnet = ResNet(Bottleneck,layers=[1,0,0,0])
        self.unet_1 = UNet(1, 73).to('cuda:0')
        self.unet_2 = UNet(1, 73).to('cuda:1')
        self.unet_3 = UNet(1, 73).to('cuda:2')
        self.unet_4 = UNet(1, 73).to('cuda:3')
        self.unet_5 = UNet(1, 73).to('cuda:4')
        self.unet_1d = UNet_1d(73*6,2,bilinear=False).to('cuda:5')
        self.conv1x1 = conv1x1(2048, 2)
        # self.attention = Attention(73*6)
        self.attention = MultiheadAttention(embed_dim=73*6,num_heads=6).to('cuda:5')

        self.classifier = nn.Linear(12, 2).to('cuda:5')

    def forward(self, x,x_1d):
        # out_1 = self.resnet(x)
        out_1 = self.unet_1(x.to('cuda:0'))
        out_2 = self.unet_2(x.to('cuda:1'))
        out_3 = self.unet_3(x.to('cuda:2'))
        out_4 = self.unet_4(x.to('cuda:3'))
        out_5 = self.unet_5(x.to('cuda:4'))
        # out_1d = self.unet_1d(x_1d.to('cuda:5'))
        # out_1d = torch.unsqueeze(out_1d,dim=-1)
        # x_1d = torch.unsqueeze(x_1d,dim=-1)
        x_1d = torch.transpose(x_1d,dim0=1,dim1=3)
        x_1d = torch.transpose(x_1d,dim0=2,dim1=3)
        # print(out_1.size(), x_1d.size())
        # out_2 = self.conv1x1(out_1)
        # out_1 = F.con
        out_1 = F.adaptive_max_pool2d(out_1, output_size=(x.size()[-2], 1))
        out_2 = F.adaptive_max_pool2d(out_2, output_size=(x.size()[-2], 1))
        out_3 = F.adaptive_max_pool2d(out_3, output_size=(x.size()[-2], 1))
        out_4 = F.adaptive_max_pool2d(out_4, output_size=(x.size()[-2], 1))
        out_5 = F.adaptive_max_pool2d(out_5, output_size=(x.size()[-2], 1))
        # print(out_1.size(), x_1d.size())
        out = torch.cat((out_1.to('cuda:5'), out_2.to('cuda:5'), out_3.to('cuda:5'), out_4.to('cuda:5'), out_5.to('cuda:5'),x_1d.to('cuda:5')), dim=1)
        # out = torch.transpose(out, 1, -1)
        # out = self.classifier(out.to('cuda:5'))
        # out = torch.transpose(out, 1, -1)
        out = torch.squeeze(out,dim=-1)
        out = torch.transpose(out,dim0=1,dim1=2)
        out = torch.transpose(out, dim0=0, dim1=1)
        out ,_= self.attention(out.to('cuda:5'),out.to('cuda:5'), out.to('cuda:5'))
        out = torch.transpose(out,dim0=0,dim1=1)
        out = torch.transpose(out, dim0=1, dim1=2)
        out = self.unet_1d(out.to('cuda:5'))
        return out


class Net_2(nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()
        # self.resnet = ResNet(Bottleneck,layers=[1,0,0,0])
        self.unet_1 = UNet(1, 1).to('cuda:0')
        # self.unet_2 = UNet(1, 73).to('cuda:1')
        # self.unet_3 = UNet(1, 73).to('cuda:2')
        # self.unet_4 = UNet(1, 73).to('cuda:3')
        # self.unet_5 = UNet(1, 73).to('cuda:4')
        self.unet_1d = UNet_1d(73,1,bilinear=False).to('cuda:1')
        self.conv1x1 = conv1x1(2048, 2)
        # self.attention = Attention(73*6)
        self.attention = MultiheadAttention(embed_dim=2,num_heads=2).to('cuda:2')

        self.classifier = nn.Linear(12, 2).to('cuda:2')

    def forward(self, x,x_1d):
        # out_1 = self.resnet(x)
        out_1 = self.unet_1(x.to('cuda:0'))
        # out_2 = self.unet_2(x.to('cuda:1'))
        # out_3 = self.unet_3(x.to('cuda:2'))
        # out_4 = self.unet_4(x.to('cuda:3'))
        # out_5 = self.unet_5(x.to('cuda:4'))
        # out_1d = self.unet_1d(x_1d.to('cuda:5'))
        # out_1d = torch.unsqueeze(out_1d,dim=-1)
        # x_1d = torch.unsqueeze(x_1d,dim=-1)
        x_1d = torch.transpose(x_1d,dim0=1,dim1=3)
        x_1d = torch.transpose(x_1d,dim0=2,dim1=3)
        x_1d = torch.squeeze(x_1d,dim=-1)
        out_1d = self.unet_1d(x_1d.to('cuda:1'))
        # print(out_1.size(), x_1d.size())
        # out_2 = self.conv1x1(out_1)
        # out_1 = F.con
        out_1 = F.adaptive_max_pool2d(out_1, output_size=(x.size()[-2], 1))
        out_1 = torch.squeeze(out_1,dim=-1)
        # out_2 = F.adaptive_max_pool2d(out_2, output_size=(x.size()[-2], 1))
        # out_3 = F.adaptive_max_pool2d(out_3, output_size=(x.size()[-2], 1))
        # out_4 = F.adaptive_max_pool2d(out_4, output_size=(x.size()[-2], 1))
        # out_5 = F.adaptive_max_pool2d(out_5, output_size=(x.size()[-2], 1))
        # print(out_1.size(), x_1d.size())
        # out = torch.cat((out_1.to('cuda:5'), out_2.to('cuda:5'), out_3.to('cuda:5'), out_4.to('cuda:5'), out_5.to('cuda:5'),x_1d.to('cuda:5')), dim=1)
        # out = torch.transpose(out, 1, -1)
        # out = self.classifier(out.to('cuda:5'))
        # out = torch.transpose(out, 1, -1)
        out = torch.cat((out_1.to('cuda:2'),out_1d.to('cuda:2')),dim=1)
        out = torch.squeeze(out,dim=-1)
        out = torch.transpose(out,dim0=1,dim1=2)
        out = torch.transpose(out, dim0=0, dim1=1)
        out ,_= self.attention(out.to('cuda:2'),out.to('cuda:2'), out.to('cuda:2'))
        out = torch.transpose(out,dim0=0,dim1=1)
        out = torch.transpose(out, dim0=1, dim1=2)
        # out = self.unet_1d(out.to('cuda:5'))
        return out


class Net_2_cpu(nn.Module):
    def __init__(self):
        super(Net_2_cpu, self).__init__()
        # self.resnet = ResNet(Bottleneck,layers=[1,0,0,0])
        self.unet_1 = UNet(1, 1)
        # self.unet_2 = UNet(1, 73).to('cuda:1')
        # self.unet_3 = UNet(1, 73).to('cuda:2')
        # self.unet_4 = UNet(1, 73).to('cuda:3')
        # self.unet_5 = UNet(1, 73).to('cuda:4')
        self.unet_1d = UNet_1d(73,1,bilinear=False)
        self.conv1x1 = conv1x1(2048, 2)
        # self.attention = Attention(73*6)
        self.attention = MultiheadAttention(embed_dim=2,num_heads=2)

        self.classifier = nn.Linear(12, 2)

    def forward(self, x,x_1d):
        # out_1 = self.resnet(x)
        out_1 = self.unet_1(x)
        # out_2 = self.unet_2(x.to('cuda:1'))
        # out_3 = self.unet_3(x.to('cuda:2'))
        # out_4 = self.unet_4(x.to('cuda:3'))
        # out_5 = self.unet_5(x.to('cuda:4'))
        # out_1d = self.unet_1d(x_1d.to('cuda:5'))
        # out_1d = torch.unsqueeze(out_1d,dim=-1)
        # x_1d = torch.unsqueeze(x_1d,dim=-1)
        x_1d = torch.transpose(x_1d,dim0=1,dim1=3)
        x_1d = torch.transpose(x_1d,dim0=2,dim1=3)
        x_1d = torch.squeeze(x_1d,dim=-1)
        out_1d = self.unet_1d(x_1d)
        # print(out_1.size(), x_1d.size())
        # out_2 = self.conv1x1(out_1)
        # out_1 = F.con
        out_1 = F.adaptive_max_pool2d(out_1, output_size=(x.size()[-2], 1))
        out_1 = torch.squeeze(out_1,dim=-1)
        # out_2 = F.adaptive_max_pool2d(out_2, output_size=(x.size()[-2], 1))
        # out_3 = F.adaptive_max_pool2d(out_3, output_size=(x.size()[-2], 1))
        # out_4 = F.adaptive_max_pool2d(out_4, output_size=(x.size()[-2], 1))
        # out_5 = F.adaptive_max_pool2d(out_5, output_size=(x.size()[-2], 1))
        # print(out_1.size(), x_1d.size())
        # out = torch.cat((out_1.to('cuda:5'), out_2.to('cuda:5'), out_3.to('cuda:5'), out_4.to('cuda:5'), out_5.to('cuda:5'),x_1d.to('cuda:5')), dim=1)
        # out = torch.transpose(out, 1, -1)
        # out = self.classifier(out.to('cuda:5'))
        # out = torch.transpose(out, 1, -1)
        out = torch.cat((out_1,out_1d),dim=1)
        out = torch.squeeze(out,dim=-1)
        out = torch.transpose(out,dim0=1,dim1=2)
        out = torch.transpose(out, dim0=0, dim1=1)
        out ,_= self.attention(out,out, out)
        out = torch.transpose(out,dim0=0,dim1=1)
        out = torch.transpose(out, dim0=1, dim1=2)
        # out = self.unet_1d(out.to('cuda:5'))
        return out

class Net_2_1d_only(nn.Module):
    def __init__(self):
        super(Net_2_1d_only, self).__init__()
        # self.resnet = ResNet(Bottleneck,layers=[1,0,0,0])
        self.unet_1 = UNet(1, 1).to('cuda:0')
        # self.unet_2 = UNet(1, 73).to('cuda:1')
        # self.unet_3 = UNet(1, 73).to('cuda:2')
        # self.unet_4 = UNet(1, 73).to('cuda:3')
        # self.unet_5 = UNet(1, 73).to('cuda:4')
        self.unet_1d = UNet_1d(73,2,bilinear=False).to('cuda:1')
        self.conv1x1 = conv1x1(2048, 2)
        # self.attention = Attention(73*6)
        self.attention = MultiheadAttention(embed_dim=2,num_heads=2).to('cuda:2')

        self.classifier = nn.Linear(12, 2).to('cuda:2')

    def forward(self, x_1d):
        # out_1 = self.resnet(x)
        # out_1 = self.unet_1(x.to('cuda:0'))
        # out_2 = self.unet_2(x.to('cuda:1'))
        # out_3 = self.unet_3(x.to('cuda:2'))
        # out_4 = self.unet_4(x.to('cuda:3'))
        # out_5 = self.unet_5(x.to('cuda:4'))
        # out_1d = self.unet_1d(x_1d.to('cuda:5'))
        # out_1d = torch.unsqueeze(out_1d,dim=-1)
        # x_1d = torch.unsqueeze(x_1d,dim=-1)
        x_1d = torch.transpose(x_1d,dim0=1,dim1=3)
        x_1d = torch.transpose(x_1d,dim0=2,dim1=3)
        x_1d = torch.squeeze(x_1d,dim=-1)
        out_1d = self.unet_1d(x_1d.to('cuda:1'))
        # print(out_1.size(), x_1d.size())
        # out_2 = self.conv1x1(out_1)
        # out_1 = F.con
        # out_1 = F.adaptive_max_pool2d(out_1, output_size=(x.size()[-2], 1))
        # out_1 = torch.squeeze(out_1,dim=-1)
        # out_2 = F.adaptive_max_pool2d(out_2, output_size=(x.size()[-2], 1))
        # out_3 = F.adaptive_max_pool2d(out_3, output_size=(x.size()[-2], 1))
        # out_4 = F.adaptive_max_pool2d(out_4, output_size=(x.size()[-2], 1))
        # out_5 = F.adaptive_max_pool2d(out_5, output_size=(x.size()[-2], 1))
        # print(out_1.size(), x_1d.size())
        # out = torch.cat((out_1.to('cuda:5'), out_2.to('cuda:5'), out_3.to('cuda:5'), out_4.to('cuda:5'), out_5.to('cuda:5'),x_1d.to('cuda:5')), dim=1)
        # out = torch.transpose(out, 1, -1)
        # out = self.classifier(out.to('cuda:5'))
        # out = torch.transpose(out, 1, -1)
        # out = torch.cat((out_1.to('cuda:2'),out_1d.to('cuda:2')),dim=1)
        out = torch.squeeze(out_1d,dim=-1)
        out = torch.transpose(out,dim0=1,dim1=2)
        out = torch.transpose(out, dim0=0, dim1=1)
        out ,_= self.attention(out.to('cuda:2'),out.to('cuda:2'), out.to('cuda:2'))
        out = torch.transpose(out,dim0=0,dim1=1)
        out = torch.transpose(out, dim0=1, dim1=2)
        # out = self.unet_1d(out.to('cuda:5'))
        return out


class Net_2_2d_only(nn.Module):
    def __init__(self):
        super(Net_2_2d_only, self).__init__()
        # self.resnet = ResNet(Bottleneck,layers=[1,0,0,0])
        self.unet_1 = UNet(1, 2).to('cuda:0')
        # self.unet_2 = UNet(1, 73).to('cuda:1')
        # self.unet_3 = UNet(1, 73).to('cuda:2')
        # self.unet_4 = UNet(1, 73).to('cuda:3')
        # self.unet_5 = UNet(1, 73).to('cuda:4')
        self.unet_1d = UNet_1d(73,1,bilinear=False).to('cuda:1')
        self.conv1x1 = conv1x1(2048, 2)
        # self.attention = Attention(73*6)
        self.attention = MultiheadAttention(embed_dim=2,num_heads=2).to('cuda:2')

        self.classifier = nn.Linear(12, 2).to('cuda:2')

    def forward(self, x):
        # out_1 = self.resnet(x)
        out_1 = self.unet_1(x.to('cuda:0'))
        # out_2 = self.unet_2(x.to('cuda:1'))
        # out_3 = self.unet_3(x.to('cuda:2'))
        # out_4 = self.unet_4(x.to('cuda:3'))
        # out_5 = self.unet_5(x.to('cuda:4'))
        # out_1d = self.unet_1d(x_1d.to('cuda:5'))
        # out_1d = torch.unsqueeze(out_1d,dim=-1)
        # x_1d = torch.unsqueeze(x_1d,dim=-1)
        # x_1d = torch.transpose(x_1d,dim0=1,dim1=3)
        # x_1d = torch.transpose(x_1d,dim0=2,dim1=3)
        # x_1d = torch.squeeze(x_1d,dim=-1)
        # out_1d = self.unet_1d(x_1d.to('cuda:1'))
        # print(out_1.size(), x_1d.size())
        # out_2 = self.conv1x1(out_1)
        # out_1 = F.con
        out_1 = F.adaptive_max_pool2d(out_1, output_size=(x.size()[-2], 1))
        out_1 = torch.squeeze(out_1,dim=-1)
        # out_2 = F.adaptive_max_pool2d(out_2, output_size=(x.size()[-2], 1))
        # out_3 = F.adaptive_max_pool2d(out_3, output_size=(x.size()[-2], 1))
        # out_4 = F.adaptive_max_pool2d(out_4, output_size=(x.size()[-2], 1))
        # out_5 = F.adaptive_max_pool2d(out_5, output_size=(x.size()[-2], 1))
        # print(out_1.size(), x_1d.size())
        # out = torch.cat((out_1.to('cuda:5'), out_2.to('cuda:5'), out_3.to('cuda:5'), out_4.to('cuda:5'), out_5.to('cuda:5'),x_1d.to('cuda:5')), dim=1)
        # out = torch.transpose(out, 1, -1)
        # out = self.classifier(out.to('cuda:5'))
        # out = torch.transpose(out, 1, -1)
        # out = torch.cat((out_1.to('cuda:2'),out_1d.to('cuda:2')),dim=1)
        out = torch.squeeze(out_1,dim=-1)
        out = torch.transpose(out,dim0=1,dim1=2)
        out = torch.transpose(out, dim0=0, dim1=1)
        out ,_= self.attention(out.to('cuda:2'),out.to('cuda:2'), out.to('cuda:2'))
        out = torch.transpose(out,dim0=0,dim1=1)
        out = torch.transpose(out, dim0=1, dim1=2)
        # out = self.unet_1d(out.to('cuda:5'))
        return out
# model = Net_2().cuda()
#
# # input = torch.rand((1,1,50,50)).cuda()
# # input_1d = torch.rand((1,1,50,73)).cuda()
# # output = model(input,input_1d)
# # print(output.size())