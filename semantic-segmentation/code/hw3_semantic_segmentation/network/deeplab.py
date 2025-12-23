import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        # raise NotImplementedError
        # TODO Problem 2.1
        # with differenrt rate
        # dilation rate
        # print(in_channels,out_channels)
        # !! padding make sure same
        self.asppconv = nn.Sequential(
          nn.Conv2d(in_channels,out_channels,3,padding=dilation,dilation=dilation),
          nn.BatchNorm2d(out_channels),
          nn.ReLU()
        )
        
    def forward(self, x):
        # print(x.shape)
        return self.asppconv(x)
        # print(x.shape)
        # ================================================================================ #

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        # raise NotImplementedError
        # TODO Problem 2.1 
        # avgpool + cnv2d
        self.pooling = nn.Sequential(
          nn.AdaptiveAvgPool2d(1),
          nn.Conv2d(in_channels,out_channels,1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),
        )
        # ================================================================================ #

    def forward(self, x):
        # print(x.shape)
        # !! upsampling to get shape back!
        # last two
        size = (x.shape[2],x.shape[3])
        x = self.pooling(x)
        # print(x.shape)
        return nn.Upsample(size=size)(x)
        # TODO Problem 2.1
        # ================================================================================ #
        # raise NotImplementedError


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        # TODO Problem 2.1
        # ================================================================================ #
        # raise NotImplementedError
        out_channels = 256
        # 3(3x3 with rate)+1(1x1)+pooling from paper
        self.conv1 = nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),
        )
        self.asconv1= ASPPConv(in_channels,out_channels, atrous_rates[0])
        self.asconv2 = ASPPConv(in_channels, out_channels, atrous_rates[1])
        self.asconv3 = ASPPConv(in_channels, out_channels, atrous_rates[2])
        self.aspool = ASPPPooling(in_channels, out_channels)
        
        self.convall = nn.Sequential(
          nn.Conv2d((1+3+1) * out_channels, out_channels, 1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),
          nn.Dropout(0.2)
        )
        # self.
        

    def forward(self, x):
        # TODO Problem 2.1
        # ================================================================================ #
        # raise NotImplementedError
        conv1= self.conv1(x)
        asc1=self.asconv1(x)
        asc2=self.asconv2(x)
        asc3=self.asconv3(x)
        aspool = self.aspool(x)
        concate = torch.cat((conv1,asc1,asc2,asc3,aspool), dim=1)
        out = self.convall(concate)
        return out


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass
    # no need for this
    # def __init__(self):
    #     super(DeepLabV3, self).__init__()
    # def forward(self, x):



class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()
        # TODO Problem 2.2
        # The model should have the following 3 arguments
        #   in_channels: number of input channels
        #   num_classes: number of classes for prediction
        #   aspp_dilate: atrous_rates for ASPP
        #   
        # ================================================================================ #
        # raise NotImplementedError
        self.aspp = ASPP(in_channels, aspp_dilate)
        # same as plus?
        self.conv = nn.Sequential(
          nn.Conv2d(256,256,3),
          nn.BatchNorm2d(256),
          nn.ReLU(),
        )
        self.classify=nn.Conv2d(256, num_classes)
        
        self._init_weight()

    def forward(self, feature):
        # TODO Problem 2.2
        # ================================================================================ #
        # raise NotImplementedError
        # !! use out here!
        out = feature['out']
        aspp = self.aspp(out)
        aspp = self.conv(aspp)
        return self.classifier(aspp)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        # TODO Problem 2.2
        # The model should have the following 4 arguments
        #   in_channels: number of input channels
        #   low_level_channels: number of channels for project
        #   num_classes: number of classes for prediction
        #   aspp_dilate: atrous_rates for ASPP
        #   
        # ================================================================================ #
        low_level_out = 64
        aspp_out = 256

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.low_1d = nn.Sequential( 
            nn.Conv2d(low_level_channels,low_level_out,1),
            nn.BatchNorm2d(low_level_out),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(low_level_out+aspp_out,aspp_out,3),
            nn.BatchNorm2d(aspp_out),
            nn.ReLU(),
        )
        self.classify = nn.Conv2d(256, num_classes, 1)
        self._init_weight()

    def forward(self, feature):
        # TODO Problem 2.2
        # ================================================================================ #
        # raise NotImplementedError
        # !!!! feature is _dict_!!
        # use out and low_level inside!
        # print(feature)
        # print(feature.type)
        # for k, v in feature.items():
        #   print(k)
        # low_level
        # out
        # !!! not only use feature here!
        low_level = feature['low_level']
        out = feature['out']
        # print(out.shape)
        # print(low_level.shape)
        low_level_out = self.low_1d(low_level)
        # print(low_level_out.shape)
        aspp_out = self.aspp(out)
        # print(aspp_out.shape)
        size= (low_level_out.shape[2],low_level_out.shape[3])
        aspp_out = nn.Upsample(size=size)(aspp_out)
        concate =torch.cat((low_level_out,aspp_out),1)
        # print(cncate.shape)
        out = self.conv(concate)
        return self.classify(out)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
