
import torch
import os
import numpy as np
import cv2
import random
random.seed(0)


def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R


def crop_img(fov, theta, phi, images, vx, vy):
    img_combine = np.zeros(images[0].shape).astype(np.float32)

    min_theta = 10000
    weights = 0
    weights_ref = np.ones(images[0].shape)
    for i, img in enumerate(images):
        _theta = vx[i]-theta
        _phi = vy[i]-phi

        if np.absolute(360+_theta) < np.absolute(_theta):
            _theta += 360

        if np.absolute(_theta) > 90:
            continue

        im_h, im_w, _ = img.shape
        K, R = get_K_R(fov, _theta, _phi, im_h, im_w)
        homo_matrix = K@R@np.linalg.inv(K)
        img_warp1 = cv2.warpPerspective(
            img, homo_matrix, (im_w, im_h)).astype(np.float32)
        _weights = cv2.warpPerspective(
            weights_ref, homo_matrix, (im_w, im_h)).astype(np.float32)
        weights += _weights

        img_combine += img_warp1  # *255).astype(np.uint8)
    img_combine = (img_combine/(weights+1e-6)).astype(np.uint8)

    return img_combine


class GeneratedDataset_old(torch.utils.data.Dataset):
    def __init__(self, config, prompt_embd, prompt_embd_null, mode='train', random_init_degree=True):

        self.prompt_embd = prompt_embd
        self.prompt_embd_null = prompt_embd_null
        self.random_init_degree = random_init_degree
        self.image_root_dir = config['image_root_dir']

        self.data = []
        _data = []
        for image_name in sorted(os.listdir(self.image_root_dir)):
            if image_name.endswith('low_res_pred.png'):
                _data.append(os.path.join(self.image_root_dir, image_name))
                if len(_data) == 8:
                    self.data.append(_data)
                    _data = []

        train_len = int(len(self.data)*0.9)

        self.vx = [45*i for i in range(8)]
        self.vy = [0]*8
        self.fov = config['fov']
        self.rot_low_res = config['rot_low_res']
        self.resolution = config['resolution']
        self.num_views_low_res = config['num_views_low_res']
        self.num_views_high_res = config['num_views_high_res']
        self.rot_high_res = config['rot_high_res']
        self.resolution_high_res = config['resolution_high_res']

    def __len__(self):
        return len(self.data)

    def load_prompt(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        return prompt

    def __getitem__(self, idx):
        images_ori = [cv2.imread(path) for path in self.data[idx]]
        images_ori = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                      for img in images_ori]

        # , min_idx1=crop_img(90, angle, 0, images, self.vx, self.vy)
        imgs = []
        Rs = []
        degrees_low_res = []

        init_degree = 0
        for i in range(0, self.num_views_low_res):
            _degree = (init_degree+self.rot_low_res*i) % 360
            img = crop_img(
                90, _degree, 0, images_ori, self.vx, self.vy)

            img = cv2.resize(img, (self.resolution, self.resolution))
            imgs.append(img)

            K, R = get_K_R(90, _degree, 0,
                           self.resolution, self.resolution)
            Rs.append(R)
            degrees_low_res.append(_degree)

        degrees_low_res = np.stack(degrees_low_res)

        images = (np.stack(imgs).astype(np.float32)/127.5)-1

        # R_inv=np.linalg.inv(R)
        K = np.stack([K]*len(Rs)).astype(np.float32)
        R = np.stack(Rs).astype(np.float32)

        degree = random.randint(1, 359)

        imgs_high_res = []
        Rs_high_res = []
        Ks_high_res = []
        degrees = []
        imgs_low_res_cond = []
        for i in range(self.num_views_high_res):
            _degree = (degree+i*self.rot_high_res) % 360
            img_high_res = crop_img(
                90, _degree, 0, images_ori, self.vx, self.vy)
            img_high_res = cv2.resize(
                img_high_res, (self.resolution_high_res, self.resolution_high_res))
            img_low_res_cond = cv2.resize(
                img_high_res, (self.resolution_high_res//4, self.resolution_high_res//4))
            K_high_res, R_high_res = get_K_R(
                90, _degree, 0, self.resolution_high_res, self.resolution_high_res)
            Rs_high_res.append(R_high_res)
            Ks_high_res.append(K_high_res)
            imgs_high_res.append(img_high_res)
            degrees.append(_degree)
            imgs_low_res_cond.append(img_low_res_cond)

        imgs_high_res = (np.stack(imgs_high_res).astype(np.float32)/127.5)-1
        imgs_low_res_cond = (
            np.stack(imgs_low_res_cond).astype(np.float32)/127.5)-1

        Rs_high_res = np.stack(Rs_high_res)
        Ks_high_res = np.stack(Ks_high_res)
        degrees = np.stack(degrees)

        prompt_gt = []
        batch_idx = int(self.data[idx][0].split('/')[-1].split('_')[1])
        with open(os.path.join(self.image_root_dir, f'0_{batch_idx}.txt')) as f:
            for line in f.readlines():
                prompt_gt.append(line.strip())
        min_diff = 1000
        prompt = []
        for degree in degrees:
            for i, d in enumerate(self.vx):
                if abs(d-degree) < min_diff:
                    min_diff = abs(d-degree)
                    min_idx = i
            prompt.append(prompt_gt[min_idx])

        return dict(images_low_res=images, imgs_low_res_conds=imgs_low_res_cond, prompt=prompt, prompt_high_res=prompt, prompt_embd=self.prompt_embd, degrees_low_res=degrees_low_res, degrees_high_res=degrees,
                    prompt_embd_null=self.prompt_embd_null, R_low_res=R, K_low_res=K, images_high_res=imgs_high_res, K_high_res=Ks_high_res, R_high_res=Rs_high_res)


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, config, prompt_embd, prompt_embd_null, mode='train', random_init_degree=True):

        self.prompt_embd = prompt_embd
        self.prompt_embd_null = prompt_embd_null
        self.random_init_degree = random_init_degree
        self.image_root_dir = config['image_root_dir']
        self.vx = [45*i for i in range(8)]
        self.vy = [0]*8
        self.fov = config['fov']
        self.rot_low_res = config['rot_low_res']
        self.resolution = config['resolution']
        self.num_views_low_res = config['num_views_low_res']
        self.num_views_high_res = config['num_views_high_res']
        self.rot_high_res = config['rot_high_res']
        self.resolution_high_res = config['resolution_high_res']
        if 'resume_dir' in config:
            self.resume_dir = config['resume_dir']
            exist_idx = []
            for name in os.listdir(self.resume_dir):
                exist_idx.append(name)
               

        self.data = []
        for image_name in sorted(os.listdir(self.image_root_dir)):
            _data = []
            if image_name in exist_idx:
                continue
            for i in range(self.num_views_high_res):
                _data.append(os.path.join(
                    self.image_root_dir, image_name, f'{i}.png'))
            self.data.append(_data)

    def __len__(self):
        return len(self.data)

    def load_prompt(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        return prompt

    def __getitem__(self, idx):
        images_ori = [cv2.imread(path) for path in self.data[idx]]
        images_ori = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                      for img in images_ori]

        # , min_idx1=crop_img(90, angle, 0, images, self.vx, self.vy)
        imgs = []
        Rs = []
        degrees_low_res = []

        init_degree = 0
        for i in range(0, self.num_views_low_res):
            _degree = (init_degree+self.rot_low_res*i) % 360

            img = crop_img(
                90, _degree, 0, images_ori, self.vx, self.vy)

            img = cv2.resize(img, (self.resolution, self.resolution))
            imgs.append(img)

            K, R = get_K_R(90, _degree, 0,
                           self.resolution, self.resolution)
            Rs.append(R)
            degrees_low_res.append(_degree)

        degrees_low_res = np.stack(degrees_low_res)

        images = (np.stack(imgs).astype(np.float32)/127.5)-1

        # R_inv=np.linalg.inv(R)
        K = np.stack([K]*len(Rs)).astype(np.float32)
        R = np.stack(Rs).astype(np.float32)

        degree = 0

        imgs_high_res = []
        Rs_high_res = []
        Ks_high_res = []
        degrees = []
        imgs_low_res_cond = []
        for i in range(self.num_views_high_res):
            _degree = (degree+i*self.rot_high_res) % 360
            img_high_res = crop_img(
                90, _degree, 0, images_ori, self.vx, self.vy)
            img_high_res = cv2.resize(
                img_high_res, (self.resolution_high_res, self.resolution_high_res))
            img_low_res_cond = cv2.resize(
                img_high_res, (self.resolution_high_res//4, self.resolution_high_res//4))
            K_high_res, R_high_res = get_K_R(
                90, _degree, 0, self.resolution_high_res, self.resolution_high_res)
            Rs_high_res.append(R_high_res)
            Ks_high_res.append(K_high_res)
            imgs_high_res.append(img_high_res)
            degrees.append(_degree)
            imgs_low_res_cond.append(img_low_res_cond)
            cv2.imwrite(f'test.png', img_high_res)

        imgs_high_res = (np.stack(imgs_high_res).astype(np.float32)/127.5)-1
        imgs_low_res_cond = (
            np.stack(imgs_low_res_cond).astype(np.float32)/127.5)-1

        Rs_high_res = np.stack(Rs_high_res)
        Ks_high_res = np.stack(Ks_high_res)
        degrees = np.stack(degrees)

        prompt_gt = []
        prompt_path = os.path.join(
            os.path.dirname(self.data[idx][0]), 'prompt.txt')
        with open(prompt_path) as f:
            for line in f.readlines():
                prompt_gt.append(line.strip())

        prompt = []
        for degree in degrees:
            min_diff = 1000
            for i, d in enumerate(self.vx):
                if abs(d-degree) < min_diff:
                    min_diff = abs(d-degree)
                    min_idx = i
            prompt.append(prompt_gt[min_idx])

        return dict(resume_dir=self.resume_dir, image_paths=self.data[idx], images_low_res=images, imgs_low_res_conds=imgs_low_res_cond, prompt=prompt, prompt_high_res=prompt, prompt_embd=self.prompt_embd, degrees_low_res=degrees_low_res, degrees_high_res=degrees,
                    prompt_embd_null=self.prompt_embd_null, R_low_res=R, K_low_res=K, images_high_res=imgs_high_res, K_high_res=Ks_high_res, R_high_res=Rs_high_res)
