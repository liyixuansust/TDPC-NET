import torch
import torch.nn as nn
from torch import Tensor
class Conv_1x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation
    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x
class Conv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=False)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation
    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x
class Conv_1x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation
    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x 
class Conv_down(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_down, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation
    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x
class Conv_3x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=False)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation
    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x
class DConv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation, d=1):
        super(DConv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(d, d, 0), dilation=d, bias=False)
        self.norm = nn.BatchNorm3d(out_dim)
        self.act = activation
    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x
class Partial_conv3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Partial_conv3, self).__init__()
        self.dim_conv3 = in_dim // 4
        self.dim_untouched = in_dim - self.dim_conv3
        self.conv = nn.Conv3d(self.dim_conv3, self.dim_conv3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=False)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.act = activation
        self.norm = nn.BatchNorm3d(out_dim)
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.conv(x1)*(self.weight1)
        x2 = x2*(self.weight2)
        x_1 = torch.cat((x1, x2), 1)
        x_1 = self.act(self.norm(self.conv1(x_1)))
        x_1 = self.conv2(x_1)
        x_1 = x + x_1
        return x_1
class HDC_module(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(HDC_module, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        self.conv_3x3x1_1 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_2 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_3 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_1x1x1_1 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(out_dim, out_dim, activation)
        if self.in_dim > self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x3x3 = Conv_1x3x3(out_dim, out_dim, activation)

    def forward(self, x):
        x_1 = self.conv_1x1x1_1(x)
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        x2 = self.conv_3x3x1_1(x2)
        x33 = self.conv_3x3x1_2(x2 + x3)
        x4 = self.conv_3x3x1_3(x33 + x4)
        x_1 = torch.cat((x1, x2, x3, x4), dim=1)
        x_1 = self.conv_1x1x1_2(x_1)
        if self.in_dim > self.out_dim:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_1x3x3(x + x_1)
        return x_1
class PHDC_module(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(PHDC_module, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        self.conv_3x3x1_1 = Partial_conv3(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_2 = Partial_conv3(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_3 = Partial_conv3(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_1x1x1_1 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(out_dim, out_dim, activation)
        if self.in_dim > self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)
    def forward(self, x):
        x_1 = self.conv_1x1x1_1(x)
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        x2 = self.conv_3x3x1_1(x2)
        x3 = self.conv_3x3x1_2(x3 + x2)
        x4 = self.conv_3x3x1_3(x4 + x3)
        x_1 = torch.cat((x1, x2, x3, x4), dim=1)
        if self.in_dim > self.out_dim:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_1x1x1_2(x + x_1)
        return x_1
device1 = torch.device("cuda")
class AD(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(AD, self).__init__()
        self.sp = Conv_3x3x3(2, 1, activation)
        self.g = nn.AdaptiveAvgPool3d(1)
        self.m = nn.AdaptiveMaxPool3d(1)
        self.aH = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.aW = nn.AdaptiveAvgPool3d((1, None, 1))
        self.aD = nn.AdaptiveAvgPool3d((1, 1, None))
        self.mH = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.mW = nn.AdaptiveMaxPool3d((1, None, 1))
        self.mD = nn.AdaptiveMaxPool3d((1, 1, None))
        self.conv1 = Conv_1x1x1(in_dim, in_dim//8, activation)
        self.conv2 = nn.Conv3d(in_dim//8, in_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.convout = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))
        self.norm = nn.BatchNorm3d(in_dim)
        self.act = activation
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):     
        xah = self.norm(self.conv2(self.conv1(self.aH(x))))
        xaw = self.norm(self.conv2(self.conv1(self.aW(x))))
        xad = self.norm(self.conv2(self.conv1(self.aD(x))))
        xahw = xah * xaw * self.weight1
        xahd = xah * xad * self.weight2
        xawd = xaw * xad * self.weight3
        xahwd1 = xahw * xahd 
        xahwd2 = xahw * xawd 
        xahwd3 = xawd * xahd 
        xahwd = xahwd1 + xahwd2 + xahwd3
        xahwd = self.sigmoid(self.g(xahwd))      
        xmh = self.norm(self.conv2(self.conv1(self.mH(x))))
        xmw = self.norm(self.conv2(self.conv1(self.mW(x))))
        xmd = self.norm(self.conv2(self.conv1(self.mD(x))))
        xmhw = xmh * xmw * self.weight1
        xmhd = xmh * xmd * self.weight2
        xmwd = xmw * xmd * self.weight3
        xmhwd1 = xmhw * xmhd 
        xmhwd2 = xmhw * xmwd 
        xmhwd3 = xmwd * xmhd 
        xmhwd = xmhwd1 + xmhwd2 + xmhwd3
        xmhwd = self.sigmoid(self.m(xmhwd))
        xc = self.convout(x * (xmhwd + xahwd))
        avg_out = torch.mean(xc, dim=1, keepdim=True)
        max_out, _ = torch.max(xc, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.sp(out))
        xo = out * xc
        return xo
def hdc(image, num=2):
    x1 = torch.Tensor([]).to(device1)
    for i in range(num):
        for j in range(num):
            for k in range(num):
                x3 = image[:, :, k::num, i::num, j::num] 
                x3 = x3.to(x1.device)
                x1 = torch.cat((x1, x3), dim=1)
    return x1
def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation)
class TDPC_Net(nn.Module):
    def __init__(self, in_dim=4, out_dim=4, num_filters=32):
        super(TDPC_Net, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_f = num_filters
        self.activation = nn.ReLU(inplace=False)
        # down
        self.conv_3x3x3 = Conv_3x3x3(self.n_f, self.n_f, self.activation)
        self.conv_1 = HDC_module(self.n_f, self.n_f, self.activation)
        self.down_1 = Conv_down(self.n_f, self.n_f, self.activation)
        self.conv_2 = HDC_module(self.n_f, self.n_f, self.activation)
        self.down_2 = Conv_down(self.n_f, self.n_f, self.activation)
        self.conv_3 = PHDC_module(self.n_f, self.n_f, self.activation)
        self.down_3 = Conv_down(self.n_f, self.n_f, self.activation)
        # bridge
        self.bridge = PHDC_module(self.n_f, self.n_f, self.activation)
        # up
        self.up_1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) 
        self.conv_4 = PHDC_module(2*self.n_f , self.n_f, self.activation)
        self.up_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  
        self.conv_5 = HDC_module(2*self.n_f , self.n_f, self.activation)
        self.up_3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) 
        self.conv_6 = HDC_module(2*self.n_f , self.n_f, self.activation)
        self.up_4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)   
        self.out = AD(self.n_f, out_dim, self.activation)
        self.softmax = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = hdc(x)
        x = self.conv_3x3x3(x)
        x1 = self.conv_1(x)
        x = self.down_1(x1)
        x2 = self.conv_2(x)
        x = self.down_2(x2)
        x3 = self.conv_3(x)
        x = self.down_3(x3)
        x = self.bridge(x)
        x = self.up_1(x)        
        x = torch.cat((x, x3), dim=1)
        x = self.conv_4(x)
        x = self.up_2(x)        
        x = torch.cat((x, x2), dim=1)
        x = self.conv_5(x)
        x = self.up_3(x)     
        x = torch.cat((x, x1), dim=1)  
        x = self.conv_6(x)
        x = self.up_4(x) 
        x = self.softmax(x)
        return x
if __name__ == '__main__':
    device = torch.device('cuda')
    image_size = 128
    x = torch.rand((1, 4, 128, 128, 128), device=device) 
    model = TDPC_Net(in_dim=4, out_dim=4, num_filters=32).to(device)
    y = model(x)
   
    
