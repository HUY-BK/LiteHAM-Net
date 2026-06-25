import torch
import torch.nn as nn
from Model.VSS_Block import VSSBlock

class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h,w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h,1), padding = 'same', groups= dim, dilation= dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1,w), padding='same', groups=dim, dilation=dilation)

    def forward(self,x):
        return x + self.dw_h(x) + self.dw_w(x)

############ Mamba Axial Channel Attention ###############
class MACA_encoder(nn.Module):
    def __init__(self, in_c, out_c, reduction = 16):
      super().__init__()
      self.dw = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c)
      self.dw2 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c)
      self.block = VSSBlock(hidden_dim=in_c // 2)
      self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
      self.bn = nn.BatchNorm2d(in_c)
      self.act = nn.LeakyReLU(negative_slope=0.01)
      self.scale = nn.Parameter(torch.ones(1))

      self.adw1 = AxialDW(in_c//2, mixer_kernel = (3,3))
      self.adw2 = AxialDW(in_c // 2, mixer_kernel=(3, 3))
      self.adw3 = AxialDW(in_c, mixer_kernel=(7, 7))

      self.shared_mlp = nn.Sequential(
        nn.Linear(in_c, in_c// reduction),
        nn.ReLU(inplace=True),
      )
      self.linear1 = nn.Linear(in_c//reduction , in_c//2)
      self.linear2 = nn.Linear(in_c // reduction, in_c // 2)

      self.sigmoid = nn.Sigmoid()
      self.avg_pool = nn.AdaptiveAvgPool2d(1)
      self.max_pool = nn.AdaptiveMaxPool2d(1)

      self.pw = nn.Conv2d(in_c, out_c, kernel_size= 1)
      self.down = nn.MaxPool2d((2, 2))

    def forward(self, x):
      residual = x
      x = self.dw(x)
      x_1, x_2 = torch.chunk(x, 2, dim=1)
      c1 = self.adw1(x_1)
      c2 = self.adw2(x_2)
      c = c1 + c2
      c = torch.cat([self.max_pool(c), self.avg_pool(c)], dim = 1)
      c = c.view(c.size(0), -1)
      c = self.shared_mlp(c)
      c1 = self.sigmoid(self.linear1(c))
      c2 = self.sigmoid(self.linear2(c))
      c1  = c1.view(c1.size(0), c1.size(1), 1,1)
      c2 = c2.view(c2.size(0), c2.size(1), 1, 1)

      x1 = x_1.permute(0, 2, 3, 1)
      x1 = self.block(x1)
      x1 = x1.permute(0, 3, 1, 2)
      x1 = self.scale * x_1 + x1
      x1 = x1 * c1

      x2 = x_2.permute(0, 2, 3, 1)
      x2 = self.block(x2)
      x2 = x2.permute(0, 3, 1, 2)
      x2 = self.scale * x_2 + x2
      x2 = x2 * c2

      x = torch.cat([x1, x2], dim=1)
      x = self.ins_norm(x) + self.adw3(residual)
      skip = self.act(self.bn(self.dw2(x)))

      out = self.down(self.pw(skip))
      return out,skip
    
    