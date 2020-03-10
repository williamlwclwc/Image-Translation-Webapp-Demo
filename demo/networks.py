import torch
import torch.nn as nn

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim), activation]
        
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# G1, global generator
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, 
                n_downsampling=3, n_blocks=6, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks >=0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                norm_layer(ngf), activation]
        # downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf*mult*2), activation]
        # resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResBlock(ngf*mult, padding_type=padding_type, 
                            activation=activation, norm_layer=norm_layer)]
        # upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), 
                                        kernel_size=3, stride=2, 
                                        padding=1, output_padding=1),
                    norm_layer(int(ngf*mult/2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# local enhancer
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, model_global, ngf=32, n_downsample_global=3, n_blocks_gobal=9,
                n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        self.model_global = model_global.model

        # global generator model
        ngf_global = ngf * (2**n_local_enhancers)
        # model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_gobal, norm_layer).model
        model_global = [self.model_global[i] for i in range(len(self.model_global)-3)] # get rid of final conv layers
        self.model = nn.Sequential(*model_global)
        
        # local enhancer layers
        for n in range(1, n_local_enhancers+1):
            # downsample
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global*2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True)]
            # residul blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]
            
            # upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, 
                                                kernel_size=3, stride=2, padding=1, output_padding=1),
                                norm_layer(ngf_global), nn.ReLU(True)]
            # final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
    
    def forward(self, input):
        # create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))
        
        # output at coarest level
        output_prev = self.model(input_downsampled[-1])
        # build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class SpadeBlock(nn.Module):
    def __init__(self, norm_nc, label_nc=3):
        super().__init__()

        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # dimention of the intermediate embedding space, hardcoded
        nhidden = 128

        ks = 3 # kernel size
        pw = ks // 2 # padding
        self.shared_net = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)


    def forward(self, x, seg_map):
        # make input go through normalization
        normalized = self.param_free_norm(x)

        # produce scaling and bias conditioned on seg map, 
        # and get rid of batch size and channel
        seg_map = F.interpolate(seg_map, size=x.size()[2:], mode='nearest')
        actv = self.shared_net(seg_map)
        gamma = self.gamma(actv)
        beta = self.beta(actv)

        # apply scale and bias
        out = normalized * gamma + beta
        return out


from torch.nn.utils.spectral_norm import spectral_norm
class SpadeResBlock(nn.Module):
    def __init__(self, fin, fout, skip=True):
        super().__init__()
        fmiddle = min(fin, fout)
        self.skip = skip
        self.spade0 = SpadeBlock(fin)
        self.conv0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.relu0 = nn.LeakyReLU(2e-1)
        self.spade1 = SpadeBlock(fmiddle)
        self.conv1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(2e-1)
        # optional skip connections
        self.spade_skip = SpadeBlock(fin)
        self.conv_skip = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        self.relu_skip = nn.LeakyReLU(2e-1)
        # optional spectral norm
        self.conv0 = spectral_norm(self.conv0)
        self.conv1 = spectral_norm(self.conv1)
        self.conv_skip = spectral_norm(self.conv_skip)


    def forward(self, x, seg_map):
        if self.skip:
            x_skip = self.conv_skip(self.relu_skip(self.spade_skip(x, seg_map)))
        else:
            x_skip = x
        
        dx = self.conv0(self.relu0(self.spade0(x, seg_map)))
        dx = self.conv1(self.relu1(self.spade1(dx, seg_map)))

        out = x_skip + dx
        return out


import torch.nn.functional as F
class SpadeGenerator(nn.Module):
    def __init__(self, plusStyle=False):
        super().__init__()
        self.plusStyle = plusStyle
        if self.plusStyle:
            pass
        else:
            # hardcoded, took from paper
            self.fc = nn.Linear(256, 1024*2*4)
        
        self.spadeRes0 = SpadeResBlock(1024, 1024)
        self.spadeRes1 = SpadeResBlock(1024, 1024)
        self.spadeRes2 = SpadeResBlock(1024, 1024)
        self.spadeRes3 = SpadeResBlock(1024, 512)
        self.spadeRes4 = SpadeResBlock(512, 256)
        self.spadeRes5 = SpadeResBlock(256, 128)
        self.spadeRes6 = SpadeResBlock(128, 64)

        self.up = nn.Upsample(scale_factor=2)
        self.conv_final = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, input, z=None):
        seg = input

        if self.plusStyle:
            pass
        else:
            if z is None:
                z = torch.randn(input.size(0), 256,
                                dtype=torch.float32)
            x = self.fc(z)
        
        b, c, h, w = seg.size()
        # hardcoded as the paper said
        x = x.view(b, 1024, 2, 4)
        x = self.spadeRes0(x, seg)
        x = self.up(x)
        x = self.spadeRes1(x, seg)
        x = self.up(x)
        x = self.spadeRes2(x, seg)
        x = self.up(x)
        x = self.spadeRes3(x, seg)
        x = self.up(x)
        x = self.spadeRes4(x, seg)
        x = self.up(x)
        x = self.spadeRes5(x, seg)
        x = self.up(x)
        x = self.spadeRes6(x, seg)
        x = self.up(x)
        x = self.conv_final(F.leaky_relu(x, 2e-1))
        out = torch.tanh(x)
        return out