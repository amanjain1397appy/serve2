import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from stylegan2_clean_arch import (ResBlock, StyleGAN2GeneratorCSFT)


class GFPGANv1Clean(nn.Module):
    """GFPGANv1 Clean version."""

    def __init__(
            self,
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True):

        super(GFPGANv1Clean, self).__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2**(int(math.log(out_size, 2)))

        self.conv_body_first = nn.Conv2d(3, channels[f'{first_out_size}'], 1)

        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(
                ResBlock(in_channels, out_channels, mode='down'))
            in_channels = out_channels

        self.final_conv = nn.Conv2d(in_channels, channels['4'], 3, 1, 1)

        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(
                ResBlock(in_channels, out_channels, mode='up'))
            in_channels = out_channels

        # to RGB
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(nn.Conv2d(channels[f'{2**i}'], 3, 1))

        if different_w:
            linear_out_channel = (int(math.log(out_size, 2))
                                  * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        self.final_linear = nn.Linear(
            channels['4'] * 4 * 4, linear_out_channel)

        self.stylegan_decoder = StyleGAN2GeneratorCSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
            sft_half=sft_half)

        if decoder_load_path:
            self.stylegan_decoder.load_state_dict(
                torch.load(decoder_load_path, map_location=lambda storage, loc: storage)['params_ema'])
        if fix_decoder:
            for name, param in self.stylegan_decoder.named_parameters():
                param.requires_grad = False

        # for SFT
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3,
                              1, 1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3,
                              1, 1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))

    def forward(self,
                x,
                return_latents=False,
                save_feat_path=None,
                load_feat_path=None,
                return_rgb=True,
                randomize_noise=True):
        conditions = []
        unet_skips = []
        out_rgbs = []

        # encoder
        feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)

        # style code
        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(
                style_code.size(0), -1, self.num_style_feat)
        # decode
        for i in range(self.log_size - 2):
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            # generate scale and shift for SFT layer
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())
            # generate rgb images
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        if save_feat_path is not None:
            torch.save(conditions, save_feat_path)
        if load_feat_path is not None:
            conditions = torch.load(load_feat_path)
            conditions = [v.cuda() for v in conditions]

        # decoder
        image, _ = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=return_latents,
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=randomize_noise)

        # return image, out_rgbs
        return image
