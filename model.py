import torch.nn as nn
import torch
import numpy as np

class ResidualBlock(nn.Module):
    #Residual Block with instance normalization.
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
    def forward(self, x):
        return x + self.main(x)

class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ResidualChannelAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  #### change to max_pool
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        max_pool = self.max_pool(out).squeeze(-1).squeeze(-1)
        max_pool = self.fc1(max_pool)
        max_pool = self.relu(max_pool)
        max_pool = self.fc2(max_pool)
        max_pool = self.sigmoid(max_pool).unsqueeze(-1).unsqueeze(-1)  ######   avg_pool
        out = out * max_pool
        out += residual
        return out

class FE(nn.Module):
    def __init__(self, in_channels, conv_dim=64, repeat_num=6):
        super(FE, self).__init__()

        layers_gt = [
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
        ]
        curr_dim = conv_dim
        for i in range(repeat_num):
            layers_gt.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        layers_gt.append(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1))
        layers_gt.append(nn.BatchNorm2d(conv_dim))
        layers_gt.append(nn.ReLU())
        layers_gt.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        layers_gt.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        layers_gt.append(nn.BatchNorm2d(conv_dim))
        layers_gt.append(nn.ReLU())
        layers_gt.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        layers_gt.append(nn.Sigmoid())
        self.features_gt = nn.Sequential(*layers_gt)

        self.fc_gt = nn.Sequential(
            nn.Linear(64 * 64 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x_gt):

        out_gt = self.features_gt(x_gt)
        out_gt = out_gt.view(out_gt.size(0), -1)
        out_gt = self.fc_gt(out_gt)

        return out_gt


class RSM(nn.Module): #RSM
    """Reflection Separament Module."""
    def __init__(self, conv_dim=64, repeat_num=12):  ####repeat_num=6
        super(RSM, self).__init__()
        layers = []
        layers.append(nn.Conv2d(4, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU())
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            curr_dim = curr_dim * 2
        for i in range(repeat_num):
            layers.append(ResidualChannelAttentionBlock(in_channels=curr_dim))
        #for i in range(repeat_num):
        #    layers.append(FE(in_channels=curr_dim))
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.ReLU6())
        #self.FeatureEnhance = nn.Sequential(*layers)
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), 1, 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x) / 6

class LM(nn.Module):
    """Loss Module."""
    def __init__(self, conv_dim=64, c_dim=1, repeat_num=5):
        super(LM, self).__init__()
        layers1 = []
        # 96*144
        layers1.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers1.append(nn.ReLU6())
        layers1.append(nn.Dropout2d(p=0.2))
        curr_dim = conv_dim
        # 48*72 * 64
        layers2 = []
        layers2.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers2.append(nn.ReLU6())
        layers2.append(nn.Dropout2d(p=0.2))
        curr_dim = curr_dim * 2
        layers3 = []
        layers3.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers3.append(nn.ReLU6(inplace=True))
        layers3.append(nn.Dropout2d(p=0.2))
        curr_dim = curr_dim * 2
        layers4 = []
        layers4.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers4.append(nn.ReLU6(inplace=True))
        layers4.append(nn.Dropout2d(p=0.2))
        curr_dim = curr_dim * 2
        layers5 = []
        layers5.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers5.append(nn.ReLU6())
        layers5.append(nn.Dropout2d(p=0.2))
        curr_dim = curr_dim * 2
        layers6 = []
        layers6.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=1, padding=1, bias=False))
        layers6.append(nn.AvgPool2d(kernel_size=4, stride=2, padding=1))
        layers6.append(nn.Dropout2d(p=0.2))
        layers6.append(nn.Sigmoid())

        self.BCE = nn.BCELoss()
        self.L1LOSS = nn.L1Loss()
        self.main1 = nn.Sequential(*layers1)
        self.main2 = nn.Sequential(*layers2)
        self.main3 = nn.Sequential(*layers3)
        self.main4 = nn.Sequential(*layers4)
        self.main5 = nn.Sequential(*layers5)
        self.main6 = nn.Sequential(*layers6)

        self.linear = nn.Linear(curr_dim * 8, 1, bias=False)
        self.sg = nn.Sigmoid()

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def compute_loss(self, x1, y1, a):
        x1 = x1.view(x1.size(0), -1)
        y1 = y1.view(y1.size(0), -1)
        loss1 = torch.div(x1 + 1e-10, y1 + 1e-10).clamp_(min=0.7, max=1.0)
        loss1_mask = (loss1 < 1) & (loss1 > 0.7)
        loss1_mask = loss1_mask.float()
        tensor_a = torch.ones(loss1.size()).cuda() * a
        loss1_error1 = self.BCE((loss1 - 0.7) / 0.3, (tensor_a - 0.7) / 0.3)
        loss1_tensor = torch.mul(loss1_error1, loss1_mask)
        loss1_out = torch.div(torch.sum(loss1_tensor, 1) + 1e-10, torch.sum(loss1_mask, 1) + 1e-10).unsqueeze(1)
        return loss1_out, torch.abs(torch.mul(x1, loss1_mask)).sum()

    def forward(self, x, y, a):
        x1 = self.main1(x)
        mu = torch.mean(torch.mean(x1, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        x1 = x1 - mu
        x2 = self.main2(x1)
        mu = torch.mean(torch.mean(x2, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        x2 = x2 - mu
        x3 = self.main3(x2)
        mu = torch.mean(torch.mean(x3, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        x3 = x3 - mu
        x4 = self.main4(x3)
        mu = torch.mean(torch.mean(x4, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        x4 = x4 - mu
        x5 = self.main5(x4)
        mu = torch.mean(torch.mean(x5, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        x5 = x5 - mu
        x6 = self.main6(x5)

        y1 = self.main1(y)
        mu = torch.mean(torch.mean(y1, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        y1 = y1 - mu
        y2 = self.main2(y1)
        mu = torch.mean(torch.mean(y2, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        y2 = y2 - mu
        y3 = self.main3(y2)
        mu = torch.mean(torch.mean(y3, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        y3 = y3 - mu
        y4 = self.main4(y3)
        mu = torch.mean(torch.mean(y4, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        y4 = y4 - mu
        y5 = self.main5(y4)
        mu = torch.mean(torch.mean(y5, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        y5 = y5 - mu
        y6 = self.main6(y5)
        loss1, _ = self.compute_loss(x1, y1, a)
        loss2, _ = self.compute_loss(x2, y2, a)
        loss3, _ = self.compute_loss(x3, y3, a)
        loss4, _ = self.compute_loss(x4, y4, a)
        loss5, loss5_out = self.compute_loss(x5, y5, a)
        loss = 0.02 * loss1 + 0.08 * loss2 + 0.2 * loss3 + 0.3 * loss4 + 0.4 * loss5
        x7 = x6.view(-1, self.num_flat_features(x6))
        x8 = self.sg(self.linear(x7)) * 0.3 + 0.7
        y7 = y6.view(-1, self.num_flat_features(y6))
        y8 = self.sg(self.linear(y7)) * 0.3 + 0.7

        return x8.view(x6.size(0), -1), y8.view(x6.size(0), -1), loss, loss5_out
"""
class FE(nn.Module):
    def __init__(self, in_channels=3, conv_dim=64, repeat_num=6):
        super(FE, self).__init__()
        layers = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(p=0.2),
        ]
        layers1 = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(p=0.2),
        ]
        layers2 = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(p=0.2),
        ]
        layers3 = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(p=0.2),
        ]
        layers4 = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(p=0.2),
        ]
        curr_dim = conv_dim
        for _ in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        layers5 = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.2),
            nn.Sigmoid(),
        ]
        #layers5.append(nn.Conv2d(conv_dim, curr_dim, kernel_size=3, padding=1))
        #layers5.append(nn.BatchNorm2d(curr_dim))
        #layers6.append(nn.ReLU6(inplace=True))
        #layers6.append(nn.Dropout2d(p=0.2))
        self.BCE = nn.BCELoss()
        self.L1Loss = nn.L1Loss()
        self.features = nn.Sequential(*layers)
        self.features = nn.Sequential(*layers1)
        self.features = nn.Sequential(*layers2)
        self.features = nn.Sequential(*layers3)
        self.features = nn.Sequential(*layers4)
        self.features = nn.Sequential(*layers5)
        self.linear = nn.Linear(curr_dim * 8, 1, bias=False)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        out = self.features(x)
        return out
"""
# can run
"""
class FE(nn.Module):
    def __init__(self, in_channels=3, conv_dim=64, repeat_num=6):
        super(FE, self).__init__()
        layers = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(p=0.2),
        ]
        layers1 = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(p=0.2),
        ]
        layers2 = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(p=0.2),
        ]
        layers3 = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(p=0.2),
        ]
        layers4 = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(p=0.2),
        ]
        curr_dim = conv_dim
        for _ in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        layers5 = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            #nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.2),
            nn.Sigmoid(),
        ]
        self.BCE = nn.BCELoss()
        self.L1Loss = nn.L1Loss()
        self.features = nn.Sequential(*layers)
        self.features = nn.Sequential(*layers1)
        self.features = nn.Sequential(*layers2)
        self.features = nn.Sequential(*layers3)
        self.features = nn.Sequential(*layers4)
        self.features = nn.Sequential(*layers5)
        self.linear = nn.Linear(curr_dim * 8, 1, bias=False)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        out = self.features(x)
        return out
"""







