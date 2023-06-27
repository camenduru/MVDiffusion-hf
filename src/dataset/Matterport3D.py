
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
    img_combine = np.zeros(images[0].shape).astype(np.uint8)

    min_theta = 10000
    for i, img in enumerate(images):
        _theta = vx[i]-theta
        _phi = vy[i]-phi

        if i == 2 and theta > 270:
            _theta = max(360-theta, _theta)
        if _phi == 0 and np.absolute(_theta) > 90:
            continue

        if i > 0 and i < 5 and np.absolute(_theta) < min_theta:
            min_theta = _theta
            min_idx = i

        im_h, im_w, _ = img.shape
        K, R = get_K_R(fov, _theta, _phi, im_h, im_w)
        homo_matrix = K@R@np.linalg.inv(K)
        img_warp1 = cv2.warpPerspective(img, homo_matrix, (im_w, im_h))
        if i == 0:
            img_warp1[im_h//2:] = 0
        elif i == 5:
            img_warp1[:im_h//2] = 0

        img_combine += img_warp1  # *255).astype(np.uint8)
    return img_combine, min_idx


def crop_img2(fov, theta, phi, images, vx, vy):
    img_combine = np.zeros(images[0].shape).astype(np.float32)

    min_theta = 10000
    weights = 0
    weights_ref = np.ones(images[0].shape)

    for i, img in enumerate(images):
        _theta = vx[i]-theta
        _phi = vy[i]-phi

        if np.absolute(360+_theta) < np.absolute(_theta):
            _theta += 360
        if np.absolute(_theta-360) < np.absolute(_theta):
            _theta -= 360

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
    img_combine = cv2.resize(img_combine, (512, 512))
    mask = ((cv2.resize(weights, (512, 512)) == 0)*255).astype(np.uint8)

    return img_combine, mask


class MP3DdatasetHomo(torch.utils.data.Dataset):
    def __init__(self, config, prompt_embd, prompt_embd_null, mode='train', random_init_degree=True):
        self.mode = mode
        self.prompt_embd = prompt_embd
        self.prompt_embd_null = prompt_embd_null
        self.random_init_degree = random_init_degree
        image_root_dir = config['image_root_dir']

        self.data = []
        for scene in sorted(os.listdir(image_root_dir)):
            img_dir = os.path.join(image_root_dir, scene,
                                   'matterport_skybox_images')
            images = []
            if os.path.exists(img_dir):
                for name in sorted(os.listdir(img_dir)):
                    if name[-3:] != 'jpg':
                        continue
                    idx = int(name.split('_')[1][-1])
                    images.append(os.path.join(img_dir, name))
                images = sorted(images)
                _images = []
                for i, image in enumerate(images):
                    _images.append(image)
                    if i % 6 == 5:
                        self.data.append(_images)
                        _images = []

        self.vx = [-90, 270, 0, 90, 180, -90]
        self.vy = [90, 0, 0, 0, 0, -90]
        self.fov = config['fov']
        self.rot_low_res = config['rot_low_res']
        self.rot_high_res = config['rot_high_res']
        self.resolution = config['resolution']
        self.resolution_high_res = config['resolution_high_res']
        self.num_views_low_res = config['num_views_low_res']
        self.num_views_high_res = config['num_views_high_res']
        self.crop_size_high_res = config['crop_size_high_res']

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

        image_idx = 1
        # , min_idx1=crop_img(90, angle, 0, images, self.vx, self.vy)
        imgs = []
        imgs_ori_reso = []
        Rs = []
        degrees_low_res = []

        if self.mode == 'train':
            init_degree = random.randint(0, 359)
        else:
            init_degree = 0
        for i in range(0, self.num_views_low_res):
            _degree = (init_degree+self.rot_low_res*i) % 360
            img, min_idx2 = crop_img(
                90, _degree, 0, images_ori, self.vx, self.vy)

            imgs_ori_reso.append(img)
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
        if self.mode == 'train':
            degree = random.randint(1, 359)
        else:
            degree = 10

        imgs_high_res = []
        Rs_high_res = []
        Ks_high_res = []
        degrees = []
        imgs_low_res_cond = []

        vx = [(init_degree+self.rot_low_res*i) %
              360 for i in range(self.num_views_low_res)]
        vy = [0]*len(vx)

        prompt_high_res = []
        image_name = self.data[idx][0].split('/')[-1].split('_')[0]
        prompt_dir = os.path.dirname(self.data[idx][0].replace(
            'matterport_skybox_images', 'blip3'))
        for i in range(self.num_views_high_res):
            _degree = (degree+i*self.rot_high_res) % 360
            img_low_res_cond, _ = crop_img2(
                90, _degree, 0, imgs_ori_reso, vx, vy)
            img_low_res_cond = cv2.resize(
                img_low_res_cond, (self.resolution_high_res//4, self.resolution_high_res//4))

            img_high_res, _ = crop_img(
                90, _degree, 0, images_ori, self.vx, self.vy)

            img_high_res = cv2.resize(
                img_high_res, (self.resolution_high_res, self.resolution_high_res))
            K_high_res, R_high_res = get_K_R(
                90, _degree, 0, self.resolution_high_res, self.resolution_high_res)
            margin = (self.resolution_high_res-self.crop_size_high_res)//2
            img_high_res = img_high_res[margin:-margin, margin:-margin]
            K_high_res[0, 2] -= margin
            K_high_res[1, 2] -= margin

            margin //= 4
            img_low_res_cond = img_low_res_cond[margin:-margin, margin:-margin]

            Rs_high_res.append(R_high_res)
            Ks_high_res.append(K_high_res)

            imgs_high_res.append(img_high_res)
            degrees.append(_degree)
            imgs_low_res_cond.append(img_low_res_cond)

            degree_prompt = int(np.round(_degree/45)*45) % 360
            txt_path = os.path.join('{}_{}.txt'.format(
                image_name, degree_prompt))

            prompt_path = os.path.join(prompt_dir, txt_path)
            prompt_high_res.append(self.load_prompt(prompt_path))

        imgs_high_res = (np.stack(imgs_high_res).astype(np.float32)/127.5)-1
        imgs_low_res_cond = (
            np.stack(imgs_low_res_cond).astype(np.float32)/127.5)-1
        Rs_high_res = np.stack(Rs_high_res)
        Ks_high_res = np.stack(Ks_high_res)
        degrees = np.stack(degrees)

        prompt = []

        for i in range(self.num_views_low_res):
            _degree = (init_degree+i*self.rot_low_res) % 360
            _degree = int(np.round(_degree/45)*45) % 360
            txt_path = os.path.join('{}_{}.txt'.format(
                image_name, _degree))

            prompt_path = os.path.join(prompt_dir, txt_path)
            prompt.append('This is one view of a scene. ' +
                          self.load_prompt(prompt_path))

        return dict(images_low_res=images, imgs_low_res_conds=imgs_low_res_cond, prompt=prompt, prompt_embd=self.prompt_embd, degrees_low_res=degrees_low_res, degrees_high_res=degrees,
                    prompt_embd_null=self.prompt_embd_null, R_low_res=R, K_low_res=K, images_high_res=imgs_high_res, K_high_res=Ks_high_res, R_high_res=Rs_high_res, prompt_high_res=prompt_high_res)


if __name__ == '__main__':
    dataset = MP3DdatasetHomo("training/mp3d_skybox", None, None)
    for i in range(10):
        dataset[i]
        import pdb
        pdb.set_trace()
