import torch
import torch.nn as nn
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))

class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=24, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.size()[2:], mode='bilinear', align_corners=True)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.size()[2:], mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.size()[2:], mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=24, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.size()[2:], mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.size()[2:], mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class U2NETLite(nn.Module):
    def __init__(self, in_ch=3, num_classes=3):
        super(U2NETLite, self).__init__()
        self.stage1 = RSU5(in_ch, 24, 32)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU5(32, 24, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU4(64, 32, 128)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(128, 64, 256)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4(256, 64, 512)
        self.stage4d = RSU4(512+256, 64, 256)
        self.stage3d = RSU4(256+128, 32, 128)
        self.stage2d = RSU5(128+64, 24, 64)
        self.stage1d = RSU5(64+32, 24, 32)
        self.side1 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.side2 = nn.Conv2d(64, num_classes, 3, padding=1)
        self.outconv = nn.Conv2d(num_classes*2, num_classes, 1)

    def forward(self, x):
        hx = x
        h1 = self.stage1(hx)
        hx = self.pool12(h1)
        h2 = self.stage2(hx)
        hx = self.pool23(h2)
        h3 = self.stage3(hx)
        hx = self.pool34(h3)
        h4 = self.stage4(hx)
        hx = self.pool45(h4)
        h5 = self.stage5(hx)
        h5_up = F.interpolate(h5, size=h4.size()[2:], mode='bilinear', align_corners=True)
        h4d = self.stage4d(torch.cat((h5_up, h4), 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=True)
        h3d = self.stage3d(torch.cat((h4dup, h3), 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=True)
        h2d = self.stage2d(torch.cat((h3dup, h2), 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=True)
        h1d = self.stage1d(torch.cat((h2dup, h1), 1))
        d1 = self.side1(h1d)
        d2 = self.side2(h2d)
        d2 = F.interpolate(d2, size=d1.size()[2:], mode='bilinear', align_corners=True)
        d0 = self.outconv(torch.cat((d1, d2), 1))
        return d0, d1

if __name__ == "__main__":
    model = U2NETLite(in_ch=3, num_classes=3).cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    d0, d1 = model(x)
    print(d0.shape, d1.shape)