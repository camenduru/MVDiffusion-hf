import torch
import torch.nn as nn
from ..modules import LREncodeNet, CorrespondenceAttn, GlobalAttn, CPAttnDecoder
from einops import rearrange
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from ..modules import GlobalAttn, CPAttnDecoder, CPBlock


class LREncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)

    def forward(self, lr_x, lr_temb=None):
        identity = lr_x
        hidden_states = self.norm1(lr_x)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        identity = hidden_states

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)+identity

        return hidden_states


class MultiViewUpsampleModel(nn.Module):
    def __init__(self, unet, config):
        super().__init__()

        self.unet = unet

        # TODO: if adding the lora layers, the validation inference does not run due to
        # shape mismatch
        self.multiframe_fuse = config['multiframe_fuse']
        if config['multiframe_fuse']:
            self.lr_conv_in = nn.Conv2d(
                in_channels=4, out_channels=self.unet.conv_in.out_channels, kernel_size=3, stride=1, padding=1)
            self.lr_downblocks = nn.ModuleList([])
            for i in range(len(self.unet.down_blocks)):
                block = LREncodingBlock(
                    in_channels=self.unet.conv_in.out_channels,
                    out_channels=self.unet.down_blocks[i].resnets[0].in_channels
                )
                self.lr_downblocks.append(block)

            self.cp_midblock_homo = CPAttnDecoder(
                self.unet.mid_block.resnets[-1].out_channels, flag360=True)

            self.cp_blocks_encoder_homo = nn.ModuleList()
            for i in range(len(self.unet.down_blocks)):
                self.cp_blocks_encoder_homo.append(CPAttnDecoder(
                    self.unet.down_blocks[i].resnets[-1].out_channels, flag360=True))

            self.cp_blocks = nn.ModuleList()
            for i in range(len(self.unet.down_blocks)):
                self.cp_blocks.append(CorrespondenceAttn(
                    self.unet.down_blocks[i].resnets[0].in_channels))

            self.cp_blocks_homo = nn.ModuleList()
            for i in range(len(self.unet.up_blocks)):
                self.cp_blocks_homo.append(CPAttnDecoder(
                    self.unet.up_blocks[i].resnets[-1].out_channels, flag360=True))

            self.trainable_parameters = [(list(self.cp_blocks.parameters()) + list(self.lr_conv_in.parameters(
            )) + list(self.lr_downblocks.parameters()) + list(self.cp_blocks_homo.parameters()) + list(self.cp_blocks_encoder_homo.parameters()), 1.0)]
        else:
            self.trainable_parameters = [(self.unet.parameters(), 1.0)]

    def forward(self, latents_hr, images_lr, timestep, prompt_embd, meta):
        if self.multiframe_fuse:
            latents_lr = meta['latents_low_res']
            m_latents_lr = latents_lr.shape[1]
            latents_lr = rearrange(latents_lr, 'b m c h w-> (b m) c h w')
            hidden_lr = self.lr_conv_in(latents_lr)
            hidden_lr = rearrange(
                hidden_lr, '(b m) c h w-> b m c h w', m=m_latents_lr)
            _, _, h_lr, w_lr = latents_lr.shape
            reso_lr = h_lr*4, w_lr*4

        deg_hr = meta['degrees_high_res']
        deg_lr = meta['degrees_low_res']
        K_hr = meta['K_high_res']
        K_lr = meta['K_low_res']
        R_hr = meta['R_high_res']
        R_lr = meta['R_low_res']
        class_labels = meta['class_labels']

        b, m, c, h, w = latents_hr.shape

        reso_hr = h*4, w*4

        m_lr = images_lr.shape[1]
        images_lr = rearrange(images_lr, 'b m c h w -> (b m) c h w')

        # bs*m, 4, 128, 128
        hidden_states = rearrange(latents_hr, 'b m c h w -> (b m) c h w')

        hidden_states = torch.cat([hidden_states, images_lr], dim=1)
        prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')

        # 1. process timesteps and class labels
        timestep = timestep.reshape(-1)
        class_labels = class_labels.reshape(-1)
        t_emb = self.unet.time_proj(timestep)  # (bs, 320)
        emb = self.unet.time_embedding(t_emb)  # (bs, 1280)
        if self.unet.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0")

            if self.unet.config.class_embed_type == "timestep":
                class_labels = self.unet.time_proj(class_labels)

            class_emb = self.unet.class_embedding(
                class_labels).to(dtype=emb.dtype)

            if self.unet.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        hidden_states = self.unet.conv_in(
            hidden_states)  # bs*m, 320, 64, 64
        # hidden_lr = self.lr_conv_in(hidden_lr)

        # unet
        # a. downsample
        down_block_res_samples = (hidden_states,)

        for i, downsample_block in enumerate(self.unet.down_blocks):
            if self.multiframe_fuse:
                hidden_lr_i = rearrange(
                    hidden_lr, 'b m c h w -> (b m) c h w', m=m_latents_lr)

                hidden_lr_i = self.lr_downblocks[i](hidden_lr_i)
                hidden_lr_i = rearrange(
                    hidden_lr_i, '(b m) c h w -> b m c h w', m=m_latents_lr)

                hidden_states = self.cp_blocks[i](
                    hidden_states, hidden_lr_i, reso_hr, reso_lr, R_hr, K_hr, R_lr, K_lr, deg_hr, deg_lr, m=m)

            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:

                for resnet, attn in zip(downsample_block.resnets, downsample_block.attentions):

                    hidden_states = resnet(hidden_states, emb)

                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample

                    down_block_res_samples += (hidden_states,)
            else:
                for resnet in downsample_block.resnets:
                    hidden_states = resnet(hidden_states, emb)
                    down_block_res_samples += (hidden_states,)
            if m > 1 and self.multiframe_fuse:
                hidden_states = self.cp_blocks_encoder_homo[i](
                    hidden_states, reso_hr, R_hr, K_hr, m)

            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)

        # b. mid
        # hidden_lr_i = self.lr_midblock(hidden_lr)
        # hidden_lr_i = rearrange(
        #    hidden_lr_i, '(b m) c h w -> b m c h w', m=m_lr)
        # hidden_states = self.cp_midblock(
        #    hidden_states, hidden_lr_i, reso_hr, reso_lr, R_hr, K_hr, R_lr, K_lr, deg_hr, deg_lr, m=m)

        hidden_states = self.unet.mid_block.resnets[0](
            hidden_states, emb)

        if m > 1 and self.multiframe_fuse:
            # hidden_states = self.global_mid_attn(hidden_states, m)
            hidden_states = self.cp_midblock_homo(
                hidden_states, reso_hr, R_hr, K_hr, m)

        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embd
            ).sample
            hidden_states = resnet(hidden_states, emb)
        # mid block epipolar cross attention
        h, w = hidden_states.shape[-2:]

        # c. upsample
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                for resnet, attn in zip(upsample_block.resnets, upsample_block.attentions):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample
            else:
                for resnet in upsample_block.resnets:
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
            if m > 1 and self.multiframe_fuse:
                hidden_states = self.cp_blocks_homo[i](
                    hidden_states, reso_hr, R_hr, K_hr, m)
            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        # 4.post-process
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)
        return sample
