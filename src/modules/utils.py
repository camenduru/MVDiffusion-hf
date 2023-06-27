import torch
import numpy as np


def get_x_2d(width, height):
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate(
        [x[..., None], y[..., None], z[..., None]], axis=-1).astype(np.float32)
    return xyz


def get_key_value(key_value, xy_l, xy_r, ori_h, ori_w, ori_h_r, ori_w_r, query_h, query_w):
    b, h, w, c = key_value.shape
    query_scale = ori_h//query_h
    key_scale = ori_h_r//h

    xy_l = xy_l[:, query_scale//2::query_scale,
                query_scale//2::query_scale]/key_scale-0.5
    key_values = []
    masks = []
    xy_floor_corres_norm = []
    xy_floor = torch.floor(xy_l)
    idx = torch.tensor(0, device=key_value.device)

    xy = get_x_2d(ori_h, ori_w)[:, :, :2]
    xy = xy[query_scale//2::query_scale, query_scale//2::query_scale]
    xy = torch.tensor(xy, device=key_value.device).float().reshape(-1, 2)

    for i in range(2):
        for j in range(2):
            _xy_floor = xy_floor.clone().reshape(b, -1, 2).long()
            _xy_floor[..., 0] += i
            _xy_floor[..., 1] += j

            mask = ((_xy_floor >= 0) & (_xy_floor < h)).all(dim=-1)
            _xy_floor = _xy_floor.clamp(min=0, max=h-1)

            _key_values = []
            for b_idx in range(_xy_floor.shape[0]):
                _key_value = key_value[b_idx,
                                       _xy_floor[b_idx, :, 1], _xy_floor[b_idx, :, 0]]
                _key_values.append(_key_value)

            _key_value = torch.stack(_key_values)

            _xy_floor_rescale = _xy_floor*key_scale
            _xy_floor_corres = []

            for k in range(b):
                _xy_floor_corres.append(
                    xy_r[k, _xy_floor_rescale[k, :, 1], _xy_floor_rescale[k, :, 0]])
            _xy_floor_corres = torch.stack(_xy_floor_corres)

            _xy_floor_corres_norm = (_xy_floor_corres-xy[None])/query_scale

            _key_value = _key_value*mask[:, :, None]
            key_values.append(_key_value)
            masks.append(mask)
            xy_floor_corres_norm.append(_xy_floor_corres_norm)

            idx += 1

    key_value = torch.stack(key_values, dim=1)
    xy_floor_corres_norm = torch.stack(xy_floor_corres_norm, dim=1)
    masks = torch.stack(masks, dim=1)
    return key_value, xy_floor_corres_norm, masks


def get_query_value(query, key_value, xy_l, xy_r, img_h_l, img_w_l, img_h_r=None, img_w_r=None):
    if img_h_r is None:
        img_h_r = img_h_l
        img_w_r = img_w_l

    b = query.shape[0]
    m = key_value.shape[1]

    key_values = []
    key_values_xy = []
    masks = []
    for i in range(m):
        _, q_h, q_w, _ = query.shape
        _key_value, _key_value_xy, mask = get_key_value(key_value[:, i], xy_l[:, i], xy_r[:, i],
                                                        img_h_l, img_w_l, img_h_r, img_w_r, q_h, q_w)

        key_values.append(_key_value)
        key_values_xy.append(_key_value_xy)
        masks.append(mask)

    key_value = torch.stack(key_values, dim=1)
    key_values_xy = torch.stack(key_values_xy, dim=1)
    mask = torch.stack(masks, dim=1)

    return query, key_value, key_values_xy, mask
