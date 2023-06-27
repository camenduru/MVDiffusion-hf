import torch
import torch.nn as nn
from einops import rearrange
from .resnet import BasicResNetBlock
from .transformer import BasicTransformerBlock, PosEmbedding
from .utils import get_x_2d, get_query_value


class LREncodeNet(nn.Module):
    def __init__(self, unet):
        super().__init__()

        self.conv_in = nn.Conv2d(
            4, 320, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        for down_block in unet.down_blocks:
            dim_out = down_block.resnets[-1].out_channels
            dim_in = down_block.resnets[0].in_channels
            blocks = BasicResNetBlock(dim_in, dim_out)
            self.down_blocks.append(blocks)

    def forward(self, x):
        outs = []
        b, m = x.shape[:2]
        x = rearrange(x, 'b m c h w -> (b m) c h w')
        x = self.conv_in(x)
        outs.append(rearrange(x, '(b m) c h w -> b m c h w', m=m))
        for block in self.down_blocks:
            x = block(x)
            outs.append(rearrange(x, '(b m) c h w -> b m c h w', m=m))

        return outs


class CorrespondenceAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = PosEmbedding(2, dim//4)

    def forward(self, x_hr, x_lr, reso_hr, reso_lr, R_hr, K_hr, R_lr, K_lr, deg_hr, deg_lr, m):

        b, c, h, w = x_hr.shape
        b = b//m
        x_hr = rearrange(x_hr, '(b m) c h w -> b m c h w', m=m)
        _, m_lr, _, h_lr, w_lr = x_lr.shape
        img_h_hr, img_w_hr = reso_hr  # h*8
        img_h_lr, img_w_lr = reso_lr
        # assert (img_h_hr == 512 and img_w_hr == 512)
        # assert (img_h_lr == 256 and img_w_lr == 256)
        out_final = []

        for b_id in range(b):
            outs = []
            for i in range(m):
                deg_diff = deg_hr[b_id, i:i+1]-deg_lr[b_id]
                mask = (torch.abs(deg_diff) < 90) | (
                    torch.abs(deg_diff-360) < 90)

                R_right = R_lr[b_id, mask]
                K_right = K_lr[b_id, mask]
                l = R_right.shape[0]

                R_left = R_hr[b_id, i:i+1].repeat(l, 1, 1)
                K_left = K_hr[b_id, i:i+1].repeat(l, 1, 1)

                R_left = R_left.reshape(-1, 3, 3)
                R_right = R_right.reshape(-1, 3, 3)
                K_left = K_left.reshape(-1, 3, 3)
                K_right = K_right.reshape(-1, 3, 3)

                homo_l = (K_right@torch.inverse(R_right) @
                          R_left@torch.inverse(K_left))
                homo_r = torch.inverse(homo_l)

                xyz_l = torch.tensor(get_x_2d(img_h_hr, img_w_hr),
                                     device=x_hr.device)
                xyz_r = torch.tensor(
                    get_x_2d(img_h_lr, img_w_lr), device=x_hr.device)

                xyz_l = (
                    xyz_l.reshape(-1, 3).T)[None].repeat(homo_l.shape[0], 1, 1)
                xyz_r = (
                    xyz_r.reshape(-1, 3).T)[None].repeat(homo_r.shape[0], 1, 1)

                xyz_l = homo_l@xyz_l  # .reshape(-1, m-1, 512, 512)
                xyz_r = homo_r@xyz_r

                xy_l = (xyz_l[:, :2]/xyz_l[:, 2:]).permute(0,
                                                           2, 1).reshape(1, l, img_h_hr, img_w_hr, 2)
                xy_r = (xyz_r[:, :2]/xyz_r[:, 2:]).permute(0, 2,
                                                           1).reshape(1, l, img_h_lr, img_w_lr, 2)

                x_left = rearrange(x_hr[b_id:b_id+1, i], 'b c h w -> b h w c')
                x_right = rearrange(
                    x_lr[b_id:b_id+1, mask], 'b m c h w -> b m h w c')

                query, key_value, key_value_xy, mask = get_query_value(
                    x_left, x_right, xy_l, xy_r, img_h_hr, img_w_hr, img_h_lr, img_w_lr)

                key_value = (key_value+self.pe(key_value_xy)) * \
                    mask[:, :, :, :, None]

                query = rearrange(query, 'b h w c->(b h w) c')[:, None]
                query_pe = self.pe(torch.zeros(
                    query.shape[0], 1, 2, device=query.device))
                key_value = rearrange(
                    key_value, 'b l m hw c -> (b hw) (l m) c')

                out = self.transformer(query, key_value, query_pe=query_pe)

                out = rearrange(out[:, 0], '(b h w) c -> b c h w', h=h, w=w)
                outs.append(out)
            out = torch.cat(outs)
            out_final.append(out)
        out_final = torch.stack(out_final)
        out_final = rearrange(out_final, 'b m c h w -> (b m) c h w')

        return out_final


class CPBlock(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.attn1 = CPAttnDecoder(dim, flag360=flag360)
        self.attn2 = CPAttnDecoder(dim, flag360=flag360)
        self.resnet = BasicResNetBlock(dim, dim, zero_init=True)

    def forward(self, x, reso, R, K, m):
        x = self.attn1(x, reso, R, K, m)
        x = self.attn2(x, reso, R, K, m)
        x = self.resnet(x)
        return x


class CPAttnDecoder(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.flag360 = flag360
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = PosEmbedding(2, dim//4)

    def forward(self, x, reso, R, K, m):
        b, c, h, w = x.shape
        x = rearrange(x, '(b m) c h w -> b m c h w', m=m)
        img_h, img_w = reso
        outs = []

        for i in range(m):
            if self.flag360:
                indexs = [(i-1+m) % m, (i+1) % m]
            else:
                if i == 0:
                    indexs = [1]
                elif i == m-1:
                    indexs = [m-2]
                else:
                    indexs = [(i-1+m) % m, (i+1) % m]

            R_right = R[:, indexs]
            K_right = K[:, indexs]
            l = R_right.shape[1]

            R_left = R[:, i:i+1].repeat(1, l, 1, 1)
            K_left = K[:, i:i+1].repeat(1, l, 1, 1)

            R_left = R_left.reshape(-1, 3, 3)
            R_right = R_right.reshape(-1, 3, 3)
            K_left = K_left.reshape(-1, 3, 3)
            K_right = K_right.reshape(-1, 3, 3)

            homo_l = (K_right@torch.inverse(R_right) @
                      R_left@torch.inverse(K_left))
            homo_r = torch.inverse(homo_l)

            xyz_l = torch.tensor(get_x_2d(img_h, img_w),
                                 device=x.device)
            xyz_r = xyz_l.clone()

            xyz_l = (
                xyz_l.reshape(-1, 3).T)[None].repeat(homo_l.shape[0], 1, 1)
            xyz_r = (
                xyz_r.reshape(-1, 3).T)[None].repeat(homo_r.shape[0], 1, 1)

            xyz_l = homo_l@xyz_l  # .reshape(-1, m-1, 512, 512)
            xyz_r = homo_r@xyz_r

            xy_l = (xyz_l[:, :2]/xyz_l[:, 2:]).permute(0,
                                                       2, 1).reshape(-1, l, img_h, img_w, 2)
            xy_r = (xyz_r[:, :2]/xyz_r[:, 2:]).permute(0, 2,
                                                       1).reshape(-1, l, img_h, img_w, 2)

            x_left = rearrange(x[:, i], 'b c h w -> b h w c')
            x_right = rearrange(
                x[:, indexs], 'b m c h w -> b m h w c')

            query, key_value, key_value_xy, mask = get_query_value(
                x_left, x_right, xy_l, xy_r, img_h, img_w, img_h, img_w)

            key_value = (key_value+self.pe(key_value_xy))*mask[..., None]

            query = rearrange(query, 'b h w c->(b h w) c')[:, None]
            query_pe = self.pe(torch.zeros(
                query.shape[0], 1, 2, device=query.device))
            key_value = rearrange(
                key_value, 'b l m hw c -> (b hw) (l m) c')

            out = self.transformer(query, key_value, query_pe=query_pe)

            out = rearrange(out[:, 0], '(b h w) c -> b c h w', h=h, w=w)
            outs.append(out)
        out = torch.stack(outs, dim=1)

        out = rearrange(out, 'b m c h w -> (b m) c h w')

        return out


class GlobalAttn(nn.Module):
    def __init__(self, channels, flag360=False):
        super().__init__()
        self.flag360 = flag360
        self.transformer = BasicTransformerBlock(
            channels, channels//32, 32, context_dim=channels, use_checkpoint=False)

    def forward(self, x, m):
        _, c, h, w = x.shape
        x = rearrange(x, '(b m) c h w -> b (m h w) c', m=m)
        x = self.transformer(x)
        x = rearrange(x, 'b (m h w) c -> (b m) c h w', h=h, m=m)
        return x
