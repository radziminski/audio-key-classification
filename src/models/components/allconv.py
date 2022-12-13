import torch.nn as nn
import torch.nn.functional as F


class AllConv(nn.Module):
    def __init__(self, num_feature_maps, dropout_rate, input_channels=1, num_classes=24):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        nf = num_feature_maps
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=nf, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=nf, out_channels=2 * nf, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=2 * nf, out_channels=2 * nf, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=2 * nf, out_channels=4 * nf, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=4 * nf, out_channels=4 * nf, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=4 * nf, out_channels=8 * nf, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=8 * nf, out_channels=8 * nf, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=8 * nf, out_channels=num_classes, kernel_size=1, padding=1)
        self.global_pool_avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p=self.dropout_rate)

        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p=self.dropout_rate)

        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p=self.dropout_rate)

        x = F.elu(self.conv7(x))
        x = F.dropout(x, p=self.dropout_rate)

        x = F.elu(self.conv8(x))
        x = F.dropout(x, p=self.dropout_rate)

        x = F.elu(self.conv9(x))
        x = self.global_pool_avg(x).squeeze(3).squeeze(2)
        return x
