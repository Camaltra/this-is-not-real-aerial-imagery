import torch
import torch.nn as nn
from ai.utils import default
from ai.models.composants import Upsample, DownSample, ConvNextBlock, SinusoidalPosEmb, BlockAttention


class OneResUNet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            sinusoidal_pos_emb_theta=10000,
            convnext_block_groups=8,
    ):
        super().__init__()
        self.channels = channels
        input_channels = channels
        self.init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, self.init_dim, 7, padding=3)

        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)

        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        DownSample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_embedding_dim=time_dim)
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_embedding_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            in_channels=dim_out + dim_in,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_out,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ConvNextBlock(dim * 2, dim, time_embedding_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        b, _, h, w = x.shape
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        unet_stack = []
        for down1, down2, downsample in self.downs:
            x = down1(x, t)
            x = down2(x, t)
            unet_stack.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for up1, up2, upsample in self.ups:
            x = torch.cat((x, unet_stack.pop()), dim=1)
            x = up1(x, t)
            x = up2(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)

        return self.final_conv(x)


class TwoResUNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        sinusoidal_pos_emb_theta=10000,
        convnext_block_groups=8,
    ):
        super().__init__()
        self.channels = channels
        input_channels = channels
        self.init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, self.init_dim, 7, padding=3)

        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)

        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        DownSample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_embedding_dim=time_dim)
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_embedding_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            is_first = ind == 0

            self.ups.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            in_channels=dim_out + dim_in,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_out + dim_in,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                    ]
                )
            )

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ConvNextBlock(dim * 2, dim, time_embedding_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        b, _, h, w = x.shape
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        unet_stack = []
        for down1, down2, downsample in self.downs:
            x = down1(x, t)
            unet_stack.append(x)
            x = down2(x, t)
            unet_stack.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for up1, up2, upsample in self.ups:
            x = torch.cat((x, unet_stack.pop()), dim=1)
            x = up1(x, t)
            x = torch.cat((x, unet_stack.pop()), dim=1)
            x = up2(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)

        return self.final_conv(x)


class AttentionUNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        sinusoidal_pos_emb_theta=10000,
        convnext_block_groups=8,
    ):
        super().__init__()
        self.channels = channels
        input_channels = channels
        self.init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, self.init_dim, 7, padding=3)

        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)

        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        DownSample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_embedding_dim=time_dim)
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_embedding_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            is_first = ind == 0

            self.ups.append(
                nn.ModuleList(
                    [
                        BlockAttention(dim_out, dim_in, 2 if not is_first else 1),
                        ConvNextBlock(
                            in_channels=dim_out + dim_in,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_out + dim_in,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                    ]
                )
            )

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ConvNextBlock(dim * 2, dim, time_embedding_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        b, _, h, w = x.shape
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        unet_stack = []
        for down1, down2, downsample in self.downs:
            x = down1(x, t)
            unet_stack.append(x)
            x = down2(x, t)
            unet_stack.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for attention, up1, up2, upsample in self.ups:
            attention_out = attention(unet_stack.pop(), x)
            x = torch.cat((x, attention_out), dim=1)
            x = up1(x, t)
            attention_out = attention(unet_stack.pop(), x)
            x = torch.cat((x, attention_out), dim=1)
            x = up2(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)

        return self.final_conv(x)


if __name__ == "__main__":
    img_in = torch.randn(1, 3, 128, 128).to("mps")
    timestamp_in = torch.randn(1).to("mps")

    models = [
        OneResUNet(64, channels=3).to("mps"),
        TwoResUNet(64, channels=3).to("mps"),
        AttentionUNet(64, channels=3).to("mps")
    ]

    for model in models:
        output = model(img_in, timestamp_in)
        assert output.shape == img_in.shape, "Not the same shape as input"
    print("Success!")
