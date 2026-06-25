
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.MACA_Encoder import *
from Model.HCAF_Bottleneck import *
from Model.MSCM import *

class CBAM(nn.Module):
    def __init__(self, in_c, reduction=16, kernel_size=3):
        super().__init__()
        self.in_c = in_c
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.channel_attn = ChannelAttention(in_c, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        chan_att = self.channel_attn(x)  # [B, C, 1, 1]
        fp = chan_att * x  # [B, C, H, W]
        spat_att = self.spatial_attn(fp)  # [B, 1, H, W]
        fpp = spat_att * fp  # [B, C, H, W]
        return fpp

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=int((kernel_size-1)/2))

    def forward(self, x):
        # Tính avg và max pooling trên chiều kênh
        max_pool = x.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        avg_pool = x.mean(dim=1, keepdim=True)    # [B, 1, H, W]
        pool = torch.cat([max_pool, avg_pool], dim=1)  # [B, 2, H, W]
        att = torch.sigmoid(self.conv(pool))  # [B, 1, H, W]
        return att

class ChannelAttention(nn.Module):
    def __init__(self, in_c, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_c, in_c // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // reduction, in_c)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)  # [B, C, 1, 1]
        max = self.max_pool(x)  # [B, C, 1, 1]

        # Làm phẳng để đưa vào MLP
        avg = avg.view(x.size(0), -1)  # [B, C]
        max = max.view(x.size(0), -1)  # [B, C]

        avg_out = self.shared_mlp(avg)  # [B, C]
        max_out = self.shared_mlp(max)  # [B, C]

        pool_sum = avg_out + max_out  # [B, C]
        sig_pool = self.sigmoid(pool_sum)  # [B, C]
        sig_pool = sig_pool.view(x.size(0), x.size(1), 1, 1)  # [B, C, 1, 1]

        return sig_pool




class AxialEncoder(nn.Module):
  def __init__(self, in_c, out_c, mixer_kernel = (7,7)):
    super().__init__()
    self.adw = AxialDW(in_c, mixer_kernel = (3,3), dilation=1)
    self.bn = nn.BatchNorm2d(in_c)
    self.act = nn.ReLU()
    self.pw  = nn.Conv2d(in_c, out_c, kernel_size = 1)
    self.down = nn.MaxPool2d((2,2))
  def forward(self, x):

    x = self.adw(x)
    x = self.act(self.bn(x))
    skip = x
    x = self.pw(x)
    x = self.down(x)
    return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + skip_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.adw = AxialDW(out_c, mixer_kernel=(3, 3))
        self.pw2 = nn.Conv2d(out_c, out_c , kernel_size = 1)

    def forward(self,x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.bn(self.pw(x))
        skip = x
        x = self.act((self.pw2(self.adw(x))))
        return x, skip

class LiteHAM_Net(nn.Module):
  def __init__(self, num_class,  c_list=[16, 32, 64, 96, 128]):
    super().__init__()
    self.conv1 = nn.Conv2d(3, c_list[0], kernel_size = 3, padding = 1)

    self.e1= AxialEncoder(c_list[0], c_list[1])
    self.e2= MACA_encoder(c_list[1], c_list[2])

    self.e3= MACA_encoder(c_list[2], c_list[3])
    self.e4= MACA_encoder(c_list[3], c_list[4])

    self.e5 = HCAF_Block(c_list[4])

    self.d4 = DecoderBlock(c_list[4], c_list[3],c_list[3])
    self.d3 = DecoderBlock(c_list[3], c_list[2], c_list[2])
    self.d2=  DecoderBlock(c_list[2], c_list[1], c_list[1])
    self.d1 = DecoderBlock(c_list[1], c_list[0], c_list[0])

    self.out = nn.Conv2d(c_list[0], num_class, kernel_size=1)

    self.cbam1 = CBAM(16, reduction=16, kernel_size = 7)
    self.cbam2 = CBAM(32, reduction=16, kernel_size = 7)
    self.cbam3 = CBAM(64, reduction=16, kernel_size = 7)
    self.cbam4 = CBAM(96, reduction=16, kernel_size = 7)

    self.pw1 = nn.Conv2d(16, 1, kernel_size=1)
    self.pw2 = nn.Conv2d(32, 1, kernel_size=1)
    self.pw3 = nn.Conv2d(64, 1, kernel_size=1)
    self.pw4 = nn.Conv2d(96, 1, kernel_size=1)

    self.mscm = MSCM_Block(c_list[1], c_list[2], c_list[3])


    self.conv_out = nn.Conv2d(4,num_class, kernel_size=1)


  def forward(self, x):
    H, W = x.shape[2:]
    x= self.conv1(x)

    x , skip1 = self.e1(x)
    x , skip2 = self.e2(x)
    x, skip3 = self.e3(x)
    x , skip4 = self.e4(x)

    x = self.e5(x)

    x1, x2,x3,x4 = self.cbam1(skip1), self.cbam2(skip2), self.cbam3(skip3), self.cbam4(skip4)

    d4, md4= self.d4(x, x4)
    d3, md3 = self.d3(d4, x3)
    d2, md2 = self.d2(d3, x2)
    d1, md1= self.d1(d2, x1)

    d2 , d3, d4= self.mscm(md2,md3,md4)


    x_in4 = F.interpolate(d4, size=(H, W), mode="bilinear", align_corners=False)
    x_in3 = F.interpolate(d3, size=(H, W), mode="bilinear", align_corners=False)
    x_in2 = F.interpolate(d2, size=(H, W), mode="bilinear", align_corners=False)
    x_in1 = F.interpolate(d1, size=(H, W), mode="bilinear", align_corners=False)


    x_in4 = self.pw4(x_in4)
    x_in3 = self.pw3(x_in3)
    x_in2 = self.pw2(x_in2)
    x_in1 = self.pw1(x_in1)

    out = torch.cat([x_in4, x_in3, x_in2, x_in1], dim = 1)
    out = self.conv_out(out)
    return out


