import torch
import torch.nn as nn
from ..modules import GlobalAttn, CPAttnDecoder, CPBlock
from einops import rearrange
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor


class MultiViewBaseModel(nn.Module):
    def __init__(self, unet, config):
        super().__init__()

        self.unet = unet
        self.single_image_ft = config['single_image_ft']

        if config['lora_layers']:
            lora_attn_procs = {}
            for name in unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith(
                    "attn1.processor") else unet.config.cross_attention_dim
                # cross_attention_dim = unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            unet.set_attn_processor(lora_attn_procs)
            lora_layers = AttnProcsLayers(unet.attn_processors)
        else:
            lora_layers = None

        if config['single_image_ft']:
            if config['lora_layers']:
                self.trainable_parameters = lora_layers.parameters()
            else:
                self.trainable_parameters = [(self.unet.parameters(), 0.01)]
        else:
            # self.global_mid_attn = GlobalAttn(
            #     self.unet.mid_block.resnets[-1].out_channels, flag360=True)

            self.cp_blocks_encoder_homo = nn.ModuleList()
            for i in range(len(self.unet.down_blocks)):
                self.cp_blocks_encoder_homo.append(CPBlock(
                    self.unet.down_blocks[i].resnets[-1].out_channels, flag360=True))

            self.cp_midblock_homo = CPBlock(
                self.unet.mid_block.resnets[-1].out_channels, flag360=True)

            self.cp_blocks_homo = nn.ModuleList()
            for i in range(len(self.unet.up_blocks)):
                self.cp_blocks_homo.append(CPBlock(
                    self.unet.up_blocks[i].resnets[-1].out_channels, flag360=True))

            self.trainable_parameters = [(list(self.cp_midblock_homo.parameters()) + \
                list(self.cp_blocks_homo.parameters()) + \
                list(self.cp_blocks_encoder_homo.parameters()), 1.0)]
          

    def forward(self, latents_hr, latents_lr, timestep, prompt_embd, meta):
        deg_hr = meta['degrees_high_res']
        deg_lr = meta['degrees_low_res']
        K_hr = meta['K_high_res']
        K_lr = meta['K_low_res']
        R_hr = meta['R_high_res']
        R_lr = meta['R_low_res']

        b, m, c, h_lr, w_lr = latents_lr.shape
        reso_lr = h_lr*8, w_lr*8

        # bs*m, 4, 64, 64
        hidden_states = rearrange(latents_lr, 'b m c h w -> (b m) c h w')
        prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')

        # 1. process timesteps

        timestep = timestep.reshape(-1)
        t_emb = self.unet.time_proj(timestep)  # (bs, 320)
        emb = self.unet.time_embedding(t_emb)  # (bs, 1280)

        hidden_states = self.unet.conv_in(
            hidden_states)  # bs*m, 320, 64, 64

        # unet
        # a. downsample
        down_block_res_samples = (hidden_states,)
        for i, downsample_block in enumerate(self.unet.down_blocks):
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
            if m > 1:
                hidden_states = self.cp_blocks_encoder_homo[i](
                    hidden_states, reso_lr, R_lr, K_lr, m)

            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)

        # b. mid

        hidden_states = self.unet.mid_block.resnets[0](
            hidden_states, emb)

        if m > 1:
            hidden_states = self.cp_midblock_homo(
                hidden_states, reso_lr, R_lr, K_lr, m)

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
            if m > 1:
                hidden_states = self.cp_blocks_homo[i](
                    hidden_states, reso_lr, R_lr, K_lr, m)

            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        # 4.post-process
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)
        return sample
