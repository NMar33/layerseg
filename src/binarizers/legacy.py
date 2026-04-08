import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Memory
from pathlib import Path


def make_padding(image, pad_size):
    h, w = image.shape[1], image.shape[2]
    h_pad, w_pad = pad_size[0], pad_size[1]
    bg = torch.zeros((1, h + 2 * h_pad, w + 2 * w_pad))
    bg[:,h_pad:h + h_pad, w_pad:w + w_pad] = image
    return bg


def _make_smart_contrast_original(image, conv_size):
    """Original loop-based implementation (kept for equivalence tests)."""
    h, w = image.shape[1], image.shape[2]
    h_pad, w_pad = int(conv_size[0] // 2), int(conv_size[1] // 2)
    h_conv, w_conv = conv_size[0], conv_size[1]
    bg = torch.zeros_like(image)
    image = make_padding(image, (h_pad, w_pad))
    for row in range(h):
        for col in range(w):
            fragment = image[:, row:(row + h_conv), col:(col + w_conv)]
            fragment_min, fragment_max = torch.min(fragment), torch.max(fragment)
            bg[:, row, col] = (fragment[:, h_pad + 1, w_pad + 1] - fragment_min) / (fragment_max - fragment_min)
    return bg


def _unfold_contrast(padded_strip, out_h, out_w, conv_size, h_pad, w_pad):
    """Vectorized contrast normalization on a single strip."""
    h_conv, w_conv = conv_size
    patches = padded_strip.unfold(1, h_conv, 1).unfold(2, w_conv, 1)
    flat = patches.reshape(1, out_h, out_w, -1)
    p_min = flat.min(dim=-1).values
    p_max = flat.max(dim=-1).values
    center = patches[:, :, :, h_pad + 1, w_pad + 1]
    return (center - p_min) / (p_max - p_min)


DEFAULT_MAX_UNFOLD_ELEMENTS = 50_000_000  # ~200 MB at float32


def make_smart_contrast(image, conv_size, max_elements=None):
    if max_elements is None:
        max_elements = DEFAULT_MAX_UNFOLD_ELEMENTS

    h, w = image.shape[1], image.shape[2]
    h_pad, w_pad = int(conv_size[0] // 2), int(conv_size[1] // 2)
    h_conv, w_conv = conv_size[0], conv_size[1]

    padded = F.pad(image.unsqueeze(0),
                   (w_pad, w_pad, h_pad, h_pad), mode='constant', value=0
                   ).squeeze(0)

    elements_per_row = w * h_conv * w_conv
    chunk_h = max(1, max_elements // elements_per_row)

    if chunk_h >= h:
        return _unfold_contrast(padded, h, w, conv_size, h_pad, w_pad)

    result = torch.zeros_like(image)
    for row_start in range(0, h, chunk_h):
        row_end = min(row_start + chunk_h, h)
        strip = padded[:, row_start:row_end + h_conv - 1, :]
        result[:, row_start:row_end, :] = \
            _unfold_contrast(strip, row_end - row_start, w, conv_size, h_pad, w_pad)
    return result


def make_img_with_smart_layers(image, max_elements=None):
    smart_contrast_layer_1 = make_smart_contrast(image, (3, 3), max_elements=max_elements)
    smart_contrast_layer_2 = make_smart_contrast(1. - image, (7, 7), max_elements=max_elements)
    image = torch.cat([image, smart_contrast_layer_1, smart_contrast_layer_2], dim=0)
    image[image.isnan()] = 0
    return image


def make_img_with_smart_layers_cached(cache_dir):
    memory = Memory(Path(cache_dir).as_posix(), verbose=0)
    make_img_with_smart_ch = memory.cache(make_img_with_smart_layers)
    return make_img_with_smart_ch


# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits