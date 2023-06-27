import pytorch_lightning as pl
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DDPMScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import math
import copy
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR
from .models import MultiViewUpsampleModel, MultiViewBaseModel


class Generator_PL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs']
        self.diff_timestep_low_res = config['model']['base_model']['diff_timestep']
        self.diff_timestep_high_res = config['model']['upsample_model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']
        self.fuse_type = config['test']['fuse_type']
        self.model_type = config['model']['model_type']

        # model_id = config['model']['model_id']

        # self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        # self.vae.eval()
        self.scheduler = DDIMScheduler.from_pretrained(
            config['model']['base_model']['model_id'], subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config['model']['base_model']['model_id'], subfolder="tokenizer", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(
            config['model']['base_model']['model_id'], subfolder="text_encoder", torch_dtype=torch.float16)
        # self.low_res_scheduler = DDIMScheduler.from_pretrained(
        #     model_id, subfolder="low_res_scheduler")
        self.low_res_noise_level = config['model']['low_res_noise_level']

        if config['model']['model_type'] == 'base' or config['model']['model_type'] == 'both':
            self.vae, self.scheduler, unet = self.load_model(
                config['model']['base_model']['model_id'])
            self.mv_base_model = MultiViewBaseModel(
                unet, config['model']['base_model'])
            self.trainable_params = self.mv_base_model.trainable_parameters
        if config['model']['model_type'] == 'upsample' or config['model']['model_type'] == 'both':
            self.vae_high_res, self.scheduler_high_res, unet = self.load_model(
                config['model']['upsample_model']['model_id'])
            self.mv_upsample_model = MultiViewUpsampleModel(
                unet, config['model']['upsample_model'])

            self.trainable_params = self.mv_upsample_model.trainable_parameters

        self.save_hyperparameters()

    def load_model(self, model_id):
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae")
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet")
        return vae, scheduler, unet

    @torch.no_grad()
    def encode_text(self, text, device):
        # text: list of str
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask)

        return prompt_embeds[0].float(), prompt_embeds[1]

    @torch.no_grad()
    def encode_image(self, x_input, vae):

        # x1 = batch['hint']
        b = x_input.shape[0]

        # x_input = torch.cat([x0.unsqueeze(1), x1.unsqueeze(1)], dim=1)
        x_input = x_input.permute(0, 1, 4, 2, 3)  # (bs, 2, 3, 512, 512)
        # (bs*2, 3, 512, 512)
        x_input = x_input.reshape(-1,
                                  x_input.shape[-3], x_input.shape[-2], x_input.shape[-1])
        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()
        z = z.reshape(b, -1, z.shape[-3], z.shape[-2],
                      z.shape[-1])  # (bs, 2, 4, 64, 64)

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        z = z.float()
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)
        # latents = rearrange(latents, 'b m c h w -> (b m) c h w')
        images = []
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)
        # image = image.reshape(latents.shape[0]//2, image.shape[-3], image.shape[-2], image.shape[-1])
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
        image = (image * 255).round().astype('uint8')

        return image

    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            param_groups.append({"params": params, "lr": self.lr * lr_scale})
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
            'interval': 'epoch',  # update the learning rate after each epoch
            'name': 'cosine_annealing_lr',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        # x: (bs, m, 4, 64, 64)  E.g. Stereo pair m=2
        # prompt_embd: (bs, 77, 1024) from SD clipText text_encoder
        # timesteps: (bs,) from ddpm schedular

        meta = {
            'degrees_high_res': batch['degrees_high_res'],
            'degrees_low_res': batch['degrees_low_res'],
            'K_high_res': batch['K_high_res'],
            'R_high_res': batch['R_high_res'],
            'R_low_res': batch['R_low_res'],
            'K_low_res': batch['K_low_res'],
            'images_high_res_debug': batch['images_high_res'],
            'images_low_res_debug': batch['images_low_res'],
        }

        if self.model_type == 'base':
            device = batch['images_low_res'].device
            prompt_embds = []
            for prompt in batch['prompt']:
                prompt_embds.append(self.encode_text(
                    prompt, device)[0])
            latents_low_res = self.encode_image(
                batch['images_low_res'], self.vae)
            t = torch.randint(0, self.scheduler.num_train_timesteps,
                              (latents_low_res.shape[0],), device=latents_low_res.device).long()
            prompt_embds = torch.stack(prompt_embds, dim=1)

            noise = torch.randn_like(latents_low_res)
            noise_z = self.scheduler.add_noise(latents_low_res, noise, t)
            t = t[:, None].repeat(1, latents_low_res.shape[1])
            denoise = self.mv_base_model(
                None, noise_z, t, prompt_embds, meta)
            target = noise
        else:
            device = batch['images_high_res'].device
            prompt_embds = []
            for prompt in batch['prompt_high_res']:
                prompt_embds.append(self.encode_text(
                    prompt, device)[0])

            latents_high_res = self.encode_image(
                batch['images_high_res'], self.vae_high_res)
            latents_low_res = self.encode_image(
                batch['images_low_res'], self.vae_high_res)

            t = torch.randint(0, self.scheduler_high_res.num_train_timesteps,
                              (latents_high_res.shape[0],), device=latents_high_res.device).long()

            prompt_embds = torch.stack(prompt_embds, dim=1)

            image_low_res = rearrange(
                batch['imgs_low_res_conds'], 'b m h w c -> b m c h w')

            s = torch.randint(0, self.scheduler.num_train_timesteps,
                              (image_low_res.shape[0],), device=device).long()
            noise_low_res = torch.randn_like(image_low_res)
            noise_z_low_res = self.scheduler.add_noise(
                image_low_res, noise_low_res, s)

            noise_latents_low_res = torch.randn_like(latents_low_res)
            noise_z_latents_low_res = self.scheduler.add_noise(
                latents_low_res, noise_latents_low_res, s)

            noise = torch.randn_like(latents_high_res)
            noise_z = self.scheduler_high_res.add_noise(
                latents_high_res, noise, t)
            t = t[:, None].repeat(1, latents_high_res.shape[1])

            meta['class_labels'] = s[:, None].repeat(
                1, latents_high_res.shape[1])
            meta['latents_low_res'] = noise_z_latents_low_res

            denoise = self.mv_upsample_model(
                noise_z, noise_z_low_res, t, prompt_embds, meta)

            latents_high_res = rearrange(
                latents_high_res, 'b l c h w -> (b l) c h w')
            noise = rearrange(noise, 'b l c h w -> (b l) c h w')
            t = rearrange(t, 'b l -> (b l)')
            denoise = rearrange(denoise, 'b l c h w -> (b l) c h w')
            target = self.scheduler_high_res.get_velocity(
                latents_high_res, noise, t)

        # eps mode
        loss = torch.nn.functional.mse_loss(denoise, target)
        self.log('train_loss', loss)
        return loss

    def gen_cls_free_guide_pair(self, latents_high_res, latents_low_res, timestep, prompt_embd, class_labels, batch):
        latents_high_res = torch.cat(
            [latents_high_res]*2) if latents_high_res is not None else None
        latents_low_res = torch.cat([latents_low_res]*2)
        timestep = torch.cat([timestep]*2)
        class_labels = torch.cat(
            [class_labels]*2) if class_labels is not None else None
        degrees_high_res = torch.cat([batch['degrees_high_res']]*2)
        degrees_low_res = torch.cat([batch['degrees_low_res']]*2)
        R_high_res = torch.cat([batch['R_high_res']]*2)
        K_high_res = torch.cat([batch['K_high_res']]*2)
        R_low_res = torch.cat([batch['R_low_res']]*2)
        K_low_res = torch.cat([batch['K_low_res']]*2)
        _latents_low_res = torch.cat(
            [batch['latents_low_res']]*2) if 'latents_low_res' in batch else None
        meta = {
            'degrees_high_res': degrees_high_res,
            'degrees_low_res': degrees_low_res,
            'K_high_res': K_high_res,
            'R_high_res': R_high_res,
            'R_low_res': R_low_res,
            'K_low_res': K_low_res,
            'class_labels': class_labels,
            'latents_low_res': _latents_low_res
        }

        return latents_high_res, latents_low_res, timestep, prompt_embd, meta

    @torch.no_grad()
    def forward_cls_free(self, latents_high_res, images_low_res, _timestep, prompt_embd, class_labels, batch, model):
        _latents_high_res, _images_low_res, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latents_high_res, images_low_res, _timestep, prompt_embd, class_labels, batch)

        noise_pred = model(
            _latents_high_res, _images_low_res, _timestep, _prompt_embd, meta)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        images_high_res_pred = None
        images_low_res_pred = None

        if self.model_type == 'upsample':
            m = batch['images_high_res'].shape[1]
            image_low_res = batch['imgs_low_res_conds']
            image_low_res = rearrange(
                image_low_res, 'b m h w c -> b m c h w')

            images_high_res_pred = self.validation_step_high_res(
                batch, image_low_res)
        elif self.model_type == 'base':
            images_low_res_pred = self.validation_step_low_res(batch)

        images_low_res = ((batch['images_low_res']/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)
        images_high_res = ((batch['images_high_res']/2+0.5) *
                           255).cpu().numpy().astype(np.uint8)
        # compute image & save
        if self.trainer.global_rank == 0:
            self.save_image(images_high_res_pred, images_low_res_pred, images_high_res,
                            images_low_res, batch['prompt'], batch_idx)

    @torch.no_grad()
    def validation_step_low_res(self, batch):
        images_low_res = batch['images_low_res']
        bs, m, h, w, _ = images_low_res.shape
        device = images_low_res.device

        latents_low_res = torch.randn(
            bs, m, 4, h//8, w//8, device=device)

        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(self.encode_text(
                prompt, device)[0])
        prompt_embds = torch.stack(prompt_embds, dim=1)

        prompt_null = batch['prompt_embd_null']
        prompt_embd = torch.cat(
            [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler.set_timesteps(self.diff_timestep_low_res, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]]*m, dim=1)

            noise_pred = self.forward_cls_free(
                None, latents_low_res, _timestep, prompt_embd, None, batch, self.mv_base_model)

            latents_low_res = self.scheduler.step(
                noise_pred, t, latents_low_res).prev_sample
        images_low_res_pred = self.decode_latent(
            latents_low_res, self.vae)
        return images_low_res_pred

    @torch.no_grad()
    def validation_step_high_res(self, batch, images_low_res):
        # x: (bs, m, 4, 64, 64)  E.g. Stereo pair m=2
        # prompt_embd: (bs, 77, 1024) from SD clipText text_encoder
        # timesteps: (bs,) from ddpm schedular

        # latents_low_res = self.encode_image(batch['images_low_res'])

        images_high_res = batch['images_high_res']
        bs, m, h, w, _ = images_high_res.shape
        device = images_high_res.device

        latents_low_res = self.encode_image(
            batch['images_low_res'], self.vae_high_res)

        batch['latents_low_res'] = latents_low_res

        # low res image prep
        # images_low_res = batch['images_low_res']
        # images_low_res = images_low_res.permute(0, 1, 4, 2, 3)
        noise_level = torch.tensor(
            [self.low_res_noise_level], dtype=torch.long, device=device)
        noise = torch.randn(images_low_res.shape,
                            device=device, dtype=images_high_res.dtype)
        images_low_res = self.scheduler.add_noise(
            images_low_res, noise, noise_level)
        noise_latents_low_res = torch.rand_like(latents_low_res)
        latents_low_res = self.scheduler.add_noise(
            latents_low_res, noise_latents_low_res, noise_level)

        latents_high_res = torch.randn(
            bs, m, 4, h//4, w//4, device=device)

        assert (bs == 1)

        prompt_embds = []
        for prompt in batch['prompt_high_res']:
            prompt_embds.append(self.encode_text(
                prompt, device)[0])
        prompt_embds = torch.stack(prompt_embds, dim=1)

        prompt_null = batch['prompt_embd_null']
        prompt_embd = torch.cat(
            [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler_high_res.set_timesteps(
            self.diff_timestep_high_res, device=device)
        timesteps = self.scheduler_high_res.timesteps

        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]]*m, dim=1)
            _noise_level = torch.cat([noise_level[:, None]]*m, dim=1)

            noise_pred = self.forward_cls_free(
                latents_high_res, images_low_res, _timestep, prompt_embd,
                _noise_level, batch, self.mv_upsample_model)

            latents_high_res = self.scheduler_high_res.step(
                noise_pred, t, latents_high_res).prev_sample

        images_high_res_pred = self.decode_latent(
            latents_high_res, self.vae_high_res)
        return images_high_res_pred

    def gen_test_homo_pair(self, latents_high_res, batch, idxs):
        batch = copy.deepcopy(batch)
        latents_high_res = latents_high_res[:, idxs]
        batch['R_high_res'] = batch['R_high_res'][:, idxs]
        batch['K_high_res'] = batch['K_high_res'][:, idxs]
        batch['degrees_high_res'] = batch['degrees_high_res'][:, idxs]

        return latents_high_res, batch

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # x: (bs, m, 4, 64, 64)  E.g. Stereo pair m=2
        # prompt_embd: (bs, 77, 1024) from SD clipText text_encoder
        # timesteps: (bs,) from ddpm schedular

        # latents_low_res=self.encode_image(batch['images_low_res'])
        images_high_res_pred = None
        images_low_res_pred = None

        if self.model_type == 'upsample':
            m = batch['images_high_res'].shape[1]
            image_low_res = batch['imgs_low_res_conds']
            image_low_res = rearrange(
                image_low_res, 'b m h w c -> b m c h w')

            images_high_res_pred = self.validation_step_high_res(
                batch, image_low_res)
        elif self.model_type == 'base':
            images_low_res_pred = self.validation_step_low_res(batch)

        images_low_res = ((batch['images_low_res']/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)
        images_high_res = ((batch['images_high_res']/2+0.5) *
                           255).cpu().numpy().astype(np.uint8)

        scene_id, image_id = batch['image_paths'][0][0].split(
            '/')[-2].split('_')
        
        output_dir = batch['resume_dir'][0] if 'resume_dir' in batch else os.path.join(self.logger.log_dir, 'images')
        output_dir=os.path.join(output_dir, "{}_{}".format(scene_id, image_id))
        
        os.makedirs(output_dir, exist_ok=True)
        for i in range(images_high_res_pred.shape[1]):
            path = os.path.join(output_dir, f'{i}.png')
            im = Image.fromarray(images_high_res_pred[0, i])
            im.save(path)
            im = Image.fromarray(images_high_res[0, i])
            path = os.path.join(output_dir, f'{i}_low_res.png')
            im.save(path)
        with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
            for p in batch['prompt']:
                f.write(p[0]+'\n')

    @torch.no_grad()
    def save_image(self, images_high_res_pred, images_low_res_pred, images_high_res, images_low_res, prompt, batch_idx):

        img_dir = os.path.join(self.logger.log_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        with open(os.path.join(img_dir, f'{self.global_step}_{batch_idx}.txt'), 'w') as f:
            for p in prompt:
                f.write(p[0]+'\n')
        if images_high_res_pred is not None:
            for m_i in range(images_high_res_pred.shape[1]):
                im = Image.fromarray(images_high_res_pred[0, m_i])
                im.save(os.path.join(
                    img_dir, f'{self.global_step}_{batch_idx}_{m_i}_high_res_pred.png'))
                if m_i < images_high_res.shape[1]:
                    im = Image.fromarray(
                        images_high_res[0, m_i])
                    im.save(os.path.join(
                        img_dir, f'{self.global_step}_{batch_idx}_{m_i}_high_res_gt.png'))
        if images_low_res_pred is not None:
            for m_i in range(images_low_res_pred.shape[1]):
                im = Image.fromarray(images_low_res_pred[0, m_i])
                im.save(os.path.join(
                    img_dir, f'{self.global_step}_{batch_idx}_{m_i}_low_res_pred.png'))

        for m_i in range(images_low_res.shape[1]):
            im = Image.fromarray(
                images_low_res[0, m_i])
            im.save(os.path.join(
                img_dir, f'{self.global_step}_{batch_idx}_{m_i}_low_res_gt.png'))
