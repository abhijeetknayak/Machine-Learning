import torch
import torch.nn as nn
import torch.functional as F

class Non_BottleNeck(nn.Module):
    def __init__(self, channels, dilation, prob):
        super().__init__()
        self.conv31_1 = nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.relu1 = nn.ReLU()
        self.conv13_1 = nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.relu2 = nn.ReLU()

        # Apply Dilation only on the second set of these filter convolutions
        self.conv31_2 = nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(dilation, 0),
                                  dilation=(dilation, 1))
        self.relu3 = nn.ReLU()
        self.conv13_2 = nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, dilation),
                                  dilation=(1, dilation))
        self.relu4 = nn.ReLU()

        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout2d(prob)

    def forward(self, X):
        out = self.conv13_1(X)
        out = self.relu1(out)

        out = self.conv13_2(out)
        out = self.relu2(out)

        out = self.conv31_1(out)
        out = self.relu3(out)

        out = self.conv31_2(out)
        out = self.relu4(out)

        out += X
        out = self.relu5(out)

        if self.dropout.p != 0.0:
            out = self.dropout(out)

        return out

class Downsampler(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c - in_c, kernel_size=(3, 3), stride=2, padding=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, X):
        conv_out = self.conv1(X)
        maxpool_out = self.maxpool(X)

        # Dim(X) = [N, C, H, W]. Use dim = 1 while concatenating
        out = torch.cat([conv_out, maxpool_out], dim=1)
        out = self.bn(out)
        out = self.relu(out)

        return out

class Upsampler(nn.Module):
    def __init__(self, in_c, out_c):
        super(Upsampler, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, padding=1, stride=2,
                                         output_padding=1, kernel_size=3)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, X):
        out = self.deconv(X)
        out = self.bn(out)

        return self.relu(out)

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(Upsampler(128, 64))
        self.layer_list.append(Non_BottleNeck(64, 1, 0.3))
        self.layer_list.append(Non_BottleNeck(64, 1, 0.3))

        self.layer_list.append(Upsampler(64, 16))
        self.layer_list.append(Non_BottleNeck(16, 1, 0.3))
        self.layer_list.append(Non_BottleNeck(16, 1, 0.3))

        self.deconv_ = nn.ConvTranspose2d(in_channels=16, out_channels=num_classes,
                                          kernel_size=3, padding=1, stride=2, output_padding=1)

    def forward(self, X):
        out = X
        for layer in self.layer_list:
            out = layer(out)

        out = self.deconv_(out)

        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.ds1 = Downsampler(in_c=3, out_c=16)
        self.ds2 = Downsampler(in_c=16, out_c=64)
        self.layer_list = nn.ModuleList()
        for i in range(5):
            self.layer_list.append(Non_BottleNeck(64, 1, 0.3))
        self.layer_list.append(Downsampler(in_c=64, out_c=128))

        for i in range(2):  # Add these layers twice
            self.layer_list.append(Non_BottleNeck(128, 2, 0.3))
            self.layer_list.append(Non_BottleNeck(128, 4, 0.3))
            self.layer_list.append(Non_BottleNeck(128, 8, 0.3))
            self.layer_list.append(Non_BottleNeck(128, 16, 0.3))

    def forward(self, X):
        out = self.ds1(X)
        out = self.ds2(out)

        for layer in self.layer_list:
            out = layer(out)

        return out

class Erfnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, X):
        out = self.encoder(X)
        out = self.decoder(out)

        return out



