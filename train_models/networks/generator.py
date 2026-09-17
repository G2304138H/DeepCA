from torch import nn
from torch import cat

from .CCT_3d import CCT

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(ConvBlock, self).__init__()

        insert_channels = out_channels #if in_channels > out_channels else out_channels // 2
        layers = [
            nn.Conv3d(in_channels, insert_channels, 3, stride=1, padding='same'),
            nn.ReLU(True),
            nn.Conv3d(insert_channels, out_channels, 3, stride=1, padding='same'),
            nn.ReLU(True)
        ]
        if batch_norm:
            layers.insert(1, nn.InstanceNorm3d(insert_channels, affine=True))
            layers.insert(len(layers)-1, nn.InstanceNorm3d(out_channels, affine=True)) 
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(Down, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)  # conv
        out = self.pool(x)  # down
        return x, out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=False):
        super(Up, self).__init__()
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest') 
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

        self.conv_block = ConvBlock(in_channels + in_channels//2, out_channels, batch_norm)

    def forward(self, x, conv):
        x = self.sample(x)  # up
        x = cat((x, conv), dim=1)  # skip connect
        x = self.conv_block(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=1, num_filters=64, class_num=1, batch_norm=True,
                 sample=False, volume_size=128):
        super(Generator, self).__init__()

        if not isinstance(volume_size, int) or isinstance(volume_size, bool):
            raise TypeError("volume_size must be an integer.")
        if volume_size < 32 or volume_size % 16 != 0:
            raise ValueError(
                "volume_size must be at least 32 and divisible by 16; "
                f"got {volume_size}."
            )
        self.volume_size = volume_size
        latent_size = volume_size // 16

        self.down1 = Down(in_channels, num_filters, batch_norm)
        self.down2 = Down(num_filters, num_filters * 2, batch_norm)
        self.down3 = Down(num_filters * 2, num_filters * 4, batch_norm)
        self.down4 = Down(num_filters * 4, num_filters * 8, batch_norm)

        self.bridge = ConvBlock(num_filters * 8, num_filters * 8, batch_norm)
        self.viTrans = CCT(vol_size=latent_size,
                           n_input_channels=num_filters * 8,
                           embedding_dim=num_filters * 8)
        self.combine = ConvBlock(num_filters * 16, num_filters * 16, batch_norm)

        self.up1 = Up(num_filters * 16, num_filters * 8, batch_norm, sample)
        self.up2 = Up(num_filters * 8, num_filters * 4, batch_norm, sample)
        self.up3 = Up(num_filters * 4, num_filters * 2, batch_norm, sample)
        self.up4 = Up(num_filters * 2, num_filters * 1, batch_norm, sample)

        self.conv_class = nn.Conv3d(num_filters * 1, class_num, 1, stride=1, padding='same')

    def forward(self, x):
        expected_shape = (self.volume_size, self.volume_size, self.volume_size)
        if x.ndim != 5 or tuple(x.shape[2:]) != expected_shape:
            raise ValueError(
                "Generator expects a 5D NCDHW tensor with cubic spatial shape "
                f"{expected_shape}, got {tuple(x.shape)}."
            )
        conv1, x = self.down1(x)
        conv2, x = self.down2(x)
        conv3, x = self.down3(x)
        conv4, x = self.down4(x)

        x_b1 = self.bridge(x)
        x_b2 = self.viTrans(x)
        x = cat([x_b1, x_b2], dim=1)
        x = self.combine(x)

        x = self.up1(x, conv4)
        x = self.up2(x, conv3)
        x = self.up3(x, conv2)
        x = self.up4(x, conv1)

        out = self.conv_class(x)

        return out
