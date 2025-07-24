import torch
import torch.nn.functional as F
from torch import nn
import math
from backbone.resnext.resnext101_regular import ResNeXt101

# CBAM
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.mean(x, 1).unsqueeze(1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels=128, reduction_ratio=16, pool_types=['avg'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

# Contrast Module (From MirrorNet)
class Contrast_Module_Deep(nn.Module):
    def __init__(self, planes, d1, d2):
        super(Contrast_Module_Deep, self).__init__()
        self.inplanes = int(planes)
        self.inplanes_half = int(planes / 2)
        self.outplanes = int(planes / 4)

        self.conv1 = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes_half, 3, 1, 1),
                                   nn.BatchNorm2d(self.inplanes_half), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(self.inplanes_half, self.outplanes, 3, 1, 1),
                                  nn.BatchNorm2d(self.outplanes), nn.ReLU())

        self.contrast_block_1 = Contrast_Block_Deep(self.outplanes, d1, d2)
        self.contrast_block_2 = Contrast_Block_Deep(self.outplanes,d1,d2)
        self.contrast_block_3 = Contrast_Block_Deep(self.outplanes,d1,d2)
        self.contrast_block_4 = Contrast_Block_Deep(self.outplanes,d1,d2)

        self.cbam = CBAM(self.inplanes)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        contrast_block_1 = self.contrast_block_1(conv2)
        contrast_block_2 = self.contrast_block_2(contrast_block_1)
        contrast_block_3 = self.contrast_block_3(contrast_block_2)
        contrast_block_4 = self.contrast_block_4(contrast_block_3)

        output = self.cbam(torch.cat((contrast_block_1, contrast_block_2, contrast_block_3, contrast_block_4), 1))

        return output

class Contrast_Block_Deep(nn.Module):
    def __init__(self, planes, d1, d2):
        super(Contrast_Block_Deep, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 2)

        self.local_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d1, dilation=d1)

        self.local_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d2, dilation=d2)

        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)


        self.relu = nn.ReLU()

        self.cbam = CBAM(self.inplanes)

    def forward(self, x):
        local_1 = self.local_1(x)
        context_1 = self.context_1(x)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn1(ccl_1)
        ccl_1 = self.relu(ccl_1)

        local_2 = self.local_2(x)
        context_2 = self.context_2(x)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn2(ccl_2)
        ccl_2 = self.relu(ccl_2)

        output = self.cbam(torch.cat((ccl_1, ccl_2), 1))

        return output

# Visual Chirality Module
class VCM(nn.Module):
    def __init__(self, planes, out_channels):
        super(VCM, self).__init__()
        d1 = 2
        d2 = 4
        d3 = 6

        self.inplanes = int(planes)
        self.outplanes = int(out_channels / 4)
        self.interplanes = int(out_channels / 2)

        self.base1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.base2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d1, dilation=d1)
        self.base3 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d2, dilation=d2)
        self.base4 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d3, dilation=d3)
        
        self.cbam_base = CBAM(self.inplanes)

        self.local1 = nn.Conv2d(self.inplanes, self.interplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context1 = nn.Conv2d(self.inplanes, self.interplanes, kernel_size=3, stride=1, padding=d2, dilation=d2)
        
        self.flip1 = nn.Conv2d(self.interplanes, self.interplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.flip2 = nn.Conv2d(self.interplanes, self.interplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        
        
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.bn2 = nn.BatchNorm2d(self.interplanes)
        self.bn3 = nn.BatchNorm2d(self.inplanes)

        self.relu = nn.ReLU()
        
        self.conv_output = nn.Sequential(nn.Conv2d(self.inplanes, out_channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))
        
    def forward(self, x):
        base1 = self.base1(x)
        base2 = self.base2(x)
        base3 = self.base3(x)
        base4 = self.base4(x)
        
        base = torch.cat((base1, base2, base3, base4), 1)
        base = self.bn1(base)
        base = self.relu(base)
        
        base = self.cbam_base(base)
        
        local1 = self.local1(x)
        context1 = self.context1(x)
        #print(context1.shape)
        flip1 = torch.flip(context1, [3])
        #print(flip1.shape)
        flip1 = self.flip1(flip1)
        flip2 = torch.flip(flip1, [3])
        #print(flip2.shape)
        flip2 = self.flip2(flip2)
        
        residual = context1 - flip2
        residual = self.bn2(residual)
        residual = self.relu(residual)
        
        chirality = torch.cat((local1, residual), 1)
        chirality = self.bn3(chirality)
        chirality = self.relu(chirality)
        
        output = self.conv_output(base * chirality)
        
        return output, residual, chirality

# GeoNet Encoder
class plainEncoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
    
        super(plainEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class resEncoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
    
        super(resEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)
        
        self.downsample = None
        if stride != 1:  
            self.downsample = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outChannel))
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out)
        return out

class EncoderNet(nn.Module):
    def __init__(self, layers):
        super(EncoderNet, self).__init__()
        
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        
        self.en_layer1 = self.make_encoder_layer(plainEncoderBlock, 64, 64, layers[0], stride=1)  
        self.en_layer2 = self.make_encoder_layer(resEncoderBlock, 64, 128, layers[1], stride=2)
        self.en_layer3 = self.make_encoder_layer(resEncoderBlock, 128, 256, layers[2], stride=2)
        self.en_layer4 = self.make_encoder_layer(resEncoderBlock, 256, 512, layers[3], stride=2)
        self.en_layer5 = self.make_encoder_layer(resEncoderBlock, 512, 512, layers[4], stride=2)
        
        # weight initializaion with Kaiming method
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def make_encoder_layer(self, block, inChannel, outChannel, block_num, stride):
        layers = []
        layers.append(block(inChannel, outChannel, stride=stride))
        for i in range(1, block_num):
            layers.append(block(outChannel, outChannel, stride=1))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = F.relu(self.bn(self.conv(x)))

        x = self.en_layer1(x)     #128
        x = self.en_layer2(x)     #64
        x = self.en_layer3(x)     #32
        x = self.en_layer4(x)     #16
        x = self.en_layer5(x)     #8
        
        return x 

# VCNet
class DNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(DNet, self).__init__()
        resnext = ResNeXt101(backbone_path)
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        # 1x1 conv
        self.attn_4 = nn.Conv2d(512, 2048, kernel_size=1)
        self.attn_3 = nn.Conv2d(512, 1024, kernel_size=1)
        self.attn_2 = nn.Conv2d(512, 512, kernel_size=1)
        self.attn_1 = nn.Conv2d(512, 256, kernel_size=1)
        
        self.edge_extract = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.chirality_edge_1_conv = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256))
        self.chirality_edge_2_conv = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256))
        self.chirality_edge_3_conv = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256))
        self.chirality_edge_4_conv = nn.Sequential(nn.Conv2d(2048, 1024, 3, 1, 1), nn.BatchNorm2d(1024), nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256))
        
        self.chirality_edge_conv = nn.Sequential(nn.Conv2d(256*4, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256))
        
        self.edge_predict = nn.Sequential(nn.Conv2d(64+512+256, 1, 3, 1, 1))


        self.contrast_4 = Contrast_Module_Deep(2048,d1=2, d2=4) # 2048x 12x12
        self.contrast_3 = Contrast_Module_Deep(1024,d1=4, d2=8) # 1024x 24x24
        self.contrast_2 = Contrast_Module_Deep(512, d1=4, d2=8) # 512x 48x48
        self.contrast_1 = Contrast_Module_Deep(256, d1=4, d2=8) # 256x 96x96

        self.ra_4 = VCM(2048, 2048)
        self.ra_3 = VCM(1024, 1024)
        self.ra_2 = VCM(512, 512)
        self.ra_1 = VCM(256, 256)

        self.up_4 = nn.Sequential(nn.ConvTranspose2d(2048, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.up_2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.up_1 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.cbam_4 = CBAM(512)
        self.cbam_3 = CBAM(256)
        self.cbam_2 = CBAM(128)
        self.cbam_1 = CBAM(64)

        self.layer4_predict = nn.Conv2d(512, 1, 3, 1, 1)
        self.layer3_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.layer2_predict = nn.Conv2d(128, 1, 3, 1, 1)
        self.layer1_predict = nn.Conv2d(64, 1, 3, 1, 1)

        self.refinement = nn.Conv2d(1+1+3+1+1+1, 1, 1, 1, 0)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x, distortion_features=None):
            layer0 = self.layer0(x)
            layer1 = self.layer1(layer0)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

            def adapter(distortion_features, layer, attn):
                _,_,h,w = layer.shape
                return torch.sigmoid(attn(F.adaptive_avg_pool2d(distortion_features,(h,w))))

            # if distortion_features is not None:
            #     layer4 = layer4 * make_attn(distortion_features, layer4, self.attn_4) # distortion_map_4
            #     layer3 = layer3 * make_attn(distortion_features, layer3, self.attn_3) # distortion_map_3
            #     layer2 = layer2 * make_attn(distortion_features, layer2, self.attn_2) # distortion_map_2
            #     layer1 = layer1 * make_attn(distortion_features, layer1, self.attn_1) # distortion_map_1
            

            # TODO: combine contrast + vc_attn with distortion map
            #TODO: up/downsample distortion map -> so that distortion map == layer4/3/2/1 spatial size
            contrast_4 = self.contrast_4(layer4) #mirrornet contextual contrasat
            cc_att_map_4, residual4, chirality_4 = self.ra_4(layer4) #vcnet
            final_contrast_4 = contrast_4 * cc_att_map_4 * adapter(distortion_features, layer4, self.attn_4) # distortion_map_4

            up_4 = self.up_4(final_contrast_4) #decoder part
            cbam_4 = self.cbam_4(up_4)
            layer4_predict = self.layer4_predict(cbam_4)
            layer4_map = torch.sigmoid(layer4_predict)

            contrast_3 = self.contrast_3(layer3 * layer4_map)
            cc_att_map_3, residual3, chirality_3 = self.ra_3(layer3 * layer4_map)
            final_contrast_3 = contrast_3 * cc_att_map_3 * adapter(distortion_features, layer3, self.attn_3) # distortion_map_3

            up_3 = self.up_3(final_contrast_3)
            cbam_3 = self.cbam_3(up_3)
            layer3_predict = self.layer3_predict(cbam_3)
            layer3_map = torch.sigmoid(layer3_predict)

            contrast_2 = self.contrast_2(layer2 * layer3_map)
            cc_att_map_2, residual2, chirality_2 = self.ra_2(layer2 * layer3_map)
            final_contrast_2 = contrast_2 * cc_att_map_2 * adapter(distortion_features, layer2, self.attn_2) # distortion_map_2

            up_2 = self.up_2(final_contrast_2)
            cbam_2 = self.cbam_2(up_2)
            layer2_predict = self.layer2_predict(cbam_2)
            layer2_map = torch.sigmoid(layer2_predict)

            contrast_1 = self.contrast_1(layer1 * layer2_map)
            cc_att_map_1, residual1, chirality_1 = self.ra_1(layer1 * layer2_map)
            final_contrast_1 = contrast_1 * cc_att_map_1 * adapter(distortion_features, layer1, self.attn_1) # distortion_map_1

            up_1 = self.up_1(final_contrast_1)
            cbam_1 = self.cbam_1(up_1)
            layer1_predict = self.layer1_predict(cbam_1)

            # edge branch (CED Module of VCNet)
            edge_feature = self.edge_extract(layer1)
            layer4_edge_feature = F.interpolate(cbam_4, size=edge_feature.size()[2:], mode='bilinear', align_corners=False)
            chirality_1_edge = self.chirality_edge_1_conv(F.interpolate(chirality_1, size=edge_feature.size()[2:], mode='bilinear', align_corners=False))
            chirality_2_edge = self.chirality_edge_2_conv(F.interpolate(chirality_2, size=edge_feature.size()[2:], mode='bilinear', align_corners=False))
            chirality_3_edge = self.chirality_edge_3_conv(F.interpolate(chirality_3, size=edge_feature.size()[2:], mode='bilinear', align_corners=False))
            chirality_4_edge = self.chirality_edge_4_conv(F.interpolate(chirality_4, size=edge_feature.size()[2:], mode='bilinear', align_corners=False))
            
            chirality = torch.cat((chirality_1_edge, chirality_2_edge, chirality_3_edge, chirality_4_edge), 1)
            chirality_edge_feature = self.chirality_edge_conv(chirality)
            
            final_edge_feature = torch.cat((edge_feature, layer4_edge_feature, chirality_edge_feature), 1)
            
            layer0_edge = self.edge_predict(final_edge_feature)

            layer4_predict = F.interpolate(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=False)
            layer3_predict = F.interpolate(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=False)
            layer2_predict = F.interpolate(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=False)
            layer1_predict = F.interpolate(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=False)

            layer0_edge = F.interpolate(layer0_edge, size=x.size()[2:], mode='bilinear', align_corners=False)

            final_features = torch.cat((x, layer1_predict, layer0_edge, layer2_predict, layer3_predict, layer4_predict), 1)
            final_predict = self.refinement(final_features)
            final_predict = F.interpolate(final_predict, size=x.size()[2:], mode='bilinear', align_corners=False)
            
            if self.training:
                return layer4_predict, layer3_predict, layer2_predict, layer1_predict, layer0_edge, final_predict
            else:
                return (torch.sigmoid(layer4_predict), torch.sigmoid(layer3_predict), 
                        torch.sigmoid(layer2_predict), torch.sigmoid(layer1_predict), 
                        torch.sigmoid(layer0_edge), torch.sigmoid(final_predict))