import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, num_classes, concat=True, output_kernel_size=1):
        super(ASPP, self).__init__()
        self.concat = concat
    
        #####ASPP
        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        ############

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)


        #####start upsample here
        kernel_size = 3
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        if self.concat:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding) # from 32x32 to 64x64
            self.upsample_2 = self.upsample(256+64, 256, 3, padding, output_padding) # from 64x64 to 128x128

        else:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding)
            self.upsample_2 = self.upsample(256, 256, 3, padding, output_padding)
            

        if output_kernel_size == 3:
            padding = 1
        elif output_kernel_size == 2:
            padding = 0
        elif output_kernel_size == 1:
            padding = 0

        self.conv_1x1_4 = nn.Conv2d(256 + 64, num_classes, kernel_size=output_kernel_size, padding=padding)

    def upsample(self, in_channels, num_filters, kernel_size, padding, output_padding):
        upsample_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels,
                                num_filters,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=output_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True)
                        )
        return upsample_layer


    def forward(self, x_high_feature, x_128=None, x_64=None, x_32=None, x_16=None):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = x_high_feature.size()[2] # (== h/16)
        feature_map_w = x_high_feature.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(x_high_feature) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))

        # need 3 times deconv, 16 -> 32, 32 -> 64, 64->128
        if self.concat:
            x = self.upsample_1(out)

            x = torch.cat([x, x_64], 1)
            x = self.upsample_2(x)
    
        else:
            x = self.upsample_1(out)
            x = self.upsample_2(x)

        x = self.conv_1x1_4(torch.cat([x, x_128], 1)) # (shape: (batch_size, num_classes, h/16, w/16))

        return x

class ASPP_Efficientnet(nn.Module):
    def __init__(self, num_classes, concat=True, output_kernel_size=1):
        super(ASPP_Efficientnet, self).__init__()
        self.concat = concat
    
        #####ASPP
        self.conv_1x1_1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(2048, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        ############

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)


        #####start upsample here
        kernel_size = 3
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        if self.concat:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding) # from 32x32 to 64x64
            self.upsample_2 = self.upsample(256+176, 256, 3, padding, output_padding) # from 64x64 to 128x128
            self.upsample_3 = self.upsample(256+64, 256, 3, padding, output_padding)
            self.upsample_4 = self.upsample(256+40, 256, 3, padding, output_padding)

        else:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding)
            self.upsample_2 = self.upsample(256, 256, 3, padding, output_padding)
            

        if output_kernel_size == 3:
            padding = 1
        elif output_kernel_size == 2:
            padding = 0
        elif output_kernel_size == 1:
            padding = 0

        self.conv_1x1_4 = nn.Conv2d(256 + 24, num_classes, kernel_size=output_kernel_size, padding=padding)

    def upsample(self, in_channels, num_filters, kernel_size, padding, output_padding):
        upsample_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels,
                                num_filters,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=output_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True)
                        )
        return upsample_layer


    def forward(self, x_high_feature, l6=None,l4=None,l3=None,l2=None):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = x_high_feature.size()[2] # (== h/16)
        feature_map_w = x_high_feature.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(x_high_feature) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        # need 3 times deconv, 16 -> 32, 32 -> 64, 64->128
        if self.concat:
            x = self.upsample_1(out)
            x = torch.cat([x, l6], 1)
            x = self.upsample_2(x)
            x = torch.cat([x, l4], 1)
            x = self.upsample_3(x)
            x = torch.cat([x, l3], 1)
            x = self.upsample_4(x)
            
        else:
            x = self.upsample_1(out)
            x = self.upsample_2(x)

        x = self.conv_1x1_4(torch.cat([x, l2], 1)) # (shape: (batch_size, num_classes, h/16, w/16))
        return x
        
class ASPP_50(nn.Module):
    def __init__(self, num_classes, concat=True, output_kernel_size=1):
        super(ASPP_50, self).__init__()
        self.concat = concat
        
        #####ASPP
        self.conv_1x1_1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(2048, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        ############

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        #####start upsample here
        kernel_size = 3
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        if self.concat:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding) # from 32x32 to 64x64
            self.upsample_2 = self.upsample(256+256, 256, 3, padding, output_padding) # from 64x64 to 128x128

        else:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding)
            self.upsample_2 = self.upsample(256, 256, 3, padding, output_padding)

        if output_kernel_size == 3:
            padding = 1
        elif output_kernel_size == 2:
            padding = 0
        elif output_kernel_size == 1:
            padding = 0

        self.conv_1x1_4 = nn.Conv2d(256 + 64, num_classes, kernel_size=output_kernel_size, padding=padding)
            

    def upsample(self, in_channels, num_filters, kernel_size, padding, output_padding):
        upsample_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels,
                                num_filters,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=output_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True)
                        )
        return upsample_layer


    def forward(self, x_high_feature, x_128=None, x_64=None, x_32=None, x_16=None):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = x_high_feature.size()[2] # (== h/16)
        feature_map_w = x_high_feature.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(x_high_feature) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))

        # need 3 times deconv, 16 -> 32, 32 -> 64, 64->128
        if self.concat:
            x = self.upsample_1(out)

            x = torch.cat([x, x_64], 1)
            x = self.upsample_2(x)
        
        else:
            x = self.upsample_1(out)
            x = self.upsample_2(x)

        x = self.conv_1x1_4(torch.cat([x, x_128], 1)) # (shape: (batch_size, num_classes, h/16, w/16))
           
        return x

class ASPP_Efficientnet_upsampled(nn.Module):
    def __init__(self, num_classes, concat=True, output_kernel_size=1):
        super(ASPP_Efficientnet_upsampled, self).__init__()
        self.concat = concat
    
        #####ASPP
        self.conv_1x1_1 = nn.Conv2d(448, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(448, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(448, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(448, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(448, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        ############

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)


        #####start upsample here
        kernel_size = 3
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        if self.concat:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding) # from 32x32 to 64x64
            self.upsample_2 = self.upsample(256+32, 256, 3, padding, output_padding) # from 64x64 to 128x128

        else:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding)
            self.upsample_2 = self.upsample(256, 256, 3, padding, output_padding)
            

        if output_kernel_size == 3:
            padding = 1
        elif output_kernel_size == 2:
            padding = 0
        elif output_kernel_size == 1:
            padding = 0

        self.conv_1x1_4 = nn.Conv2d(256 + 24, num_classes, kernel_size=output_kernel_size, padding=padding)

    def upsample(self, in_channels, num_filters, kernel_size, padding, output_padding):
        upsample_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels,
                                num_filters,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=output_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True)
                        )
        return upsample_layer


    def forward(self, x_high_feature, l3=None,l2=None):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = x_high_feature.size()[2] # (== h/16)
        feature_map_w = x_high_feature.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(x_high_feature) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        # need 3 times deconv, 16 -> 32, 32 -> 64, 64->128
        if self.concat:
            x = self.upsample_1(out)
            x = torch.cat([x, l3], 1)
            x = self.upsample_2(x)
            
        else:
            x = self.upsample_1(out)
            x = self.upsample_2(x)

        x = self.conv_1x1_4(torch.cat([x, l2], 1)) # (shape: (batch_size, num_classes, h/16, w/16))
        return x

class ASPP_Efficientnetv2(nn.Module):
    def __init__(self, num_classes, concat=True, output_kernel_size=1):
        super(ASPP_Efficientnetv2, self).__init__()
        self.concat = concat
    
        #####ASPP
        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        ############

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)


        #####start upsample here
        kernel_size = 3
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        if self.concat:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding) # from 32x32 to 64x64
            self.upsample_2 = self.upsample(256+48, 256, 3, padding, output_padding) # from 64x64 to 128x128

        else:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding)
            self.upsample_2 = self.upsample(256, 256, 3, padding, output_padding)
            

        if output_kernel_size == 3:
            padding = 1
        elif output_kernel_size == 2:
            padding = 0
        elif output_kernel_size == 1:
            padding = 0

        self.conv_1x1_4 = nn.Conv2d(256 + 24, num_classes, kernel_size=output_kernel_size, padding=padding)

    def upsample(self, in_channels, num_filters, kernel_size, padding, output_padding):
        upsample_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels,
                                num_filters,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=output_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True)
                        )
        return upsample_layer


    def forward(self, x_high_feature, l3=None,l2=None):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = x_high_feature.size()[2] # (== h/16)
        feature_map_w = x_high_feature.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(x_high_feature) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        # need 3 times deconv, 16 -> 32, 32 -> 64, 64->128
        if self.concat:
            x = self.upsample_1(out)
            x = torch.cat([x, l3], 1)
            x = self.upsample_2(x)
            
        else:
            x = self.upsample_1(out)
            x = self.upsample_2(x)

        x = self.conv_1x1_4(torch.cat([x, l2], 1)) # (shape: (batch_size, num_classes, h/16, w/16))
        return x

class ASPP_non_binary_ablationstudy(nn.Module):
    def __init__(self, code_output_length=16, divided_number_each_iteration=2, concat=True, decoder_last_layer_also_concat=False, output_kernel_size=1):
        super(ASPP_non_binary_ablationstudy, self).__init__()
        self.concat = concat
        self.decoder_last_layer_also_concat=decoder_last_layer_also_concat
        #####ASPP
        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        ############

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)


        #####start upsample here
        kernel_size = 3
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        if self.concat:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding) # from 32x32 to 64x64
            self.upsample_2 = self.upsample(256+64, 256, 3, padding, output_padding) # from 64x64 to 128x128
        
        else:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding)
            self.upsample_2 = self.upsample(256, 256, 3, padding, output_padding)
          
        if output_kernel_size == 3:
            padding = 1
        elif output_kernel_size == 2:
            padding = 0
        elif output_kernel_size == 1:
            padding = 0

        
        self.conv_1x1_4_mask = nn.Conv2d(256 + 64, 1, kernel_size=output_kernel_size, padding=padding)
        self.conv_1x1_4_code = nn.Conv2d(256 + 64, code_output_length*divided_number_each_iteration, kernel_size=output_kernel_size, padding=padding)
        

    def upsample(self, in_channels, num_filters, kernel_size, padding, output_padding):
        upsample_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels,
                                num_filters,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=output_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True)
                        )
        return upsample_layer


    def forward(self, x_high_feature, x_128=None, x_64=None, x_32=None, x_16=None):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = x_high_feature.size()[2] # (== h/16)
        feature_map_w = x_high_feature.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(x_high_feature) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))

        # need 3 times deconv, 16 -> 32, 32 -> 64, 64->128
        if self.concat:
            x = self.upsample_1(out)

            x = torch.cat([x, x_64], 1)
            x = self.upsample_2(x)
            
        else:
            x = self.upsample_1(out)
            x = self.upsample_2(x)
        
        mask = self.conv_1x1_4_mask(torch.cat([x, x_128], 1)) # (shape: (batch_size, num_classes, h/16, w/16))
        binary_code = self.conv_1x1_4_code(torch.cat([x, x_128], 1)) # (shape: (batch_size, num_classes, h/16, w/16))

        return mask, binary_code
