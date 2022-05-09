import imgaug.augmenters as iaa
import numpy as np
import cv2
import random
from os import listdir
from os.path import isfile, join

truncate_fg=True,
change_back_ground_prob=0.5
color_aug_prob=0.8


image_augmentations_lm=(
        iaa.Sequential([
                iaa.Sometimes(0.4, iaa.CoarseDropout( p=0.1, size_percent=0.05) ),
                iaa.Sometimes(0.5, iaa.GaussianBlur(np.random.rand())),
                iaa.Sometimes(0.5, iaa.Add((-20, 20), per_channel=0.3)),
                iaa.Sometimes(0.4, iaa.Invert(0.20, per_channel=True)),
                iaa.Sometimes(0.5, iaa.Multiply((0.7, 1.4), per_channel=0.8)),
                iaa.Sometimes(0.5, iaa.Multiply((0.7, 1.4))),
                iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.0), per_channel=0.3))
        ], random_order=False)
    )

image_augmentations_bop=(
        iaa.Sequential([
                iaa.Sometimes(0.5, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
                iaa.Sometimes(0.5, iaa.GaussianBlur(1.2*np.random.rand())),
                iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3)),
                iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
                iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
                iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
                iaa.Sometimes(0.5, iaa.LinearContrast((0.5, 2.2), per_channel=0.3))
                ], random_order = False)
        )

def resize_short_edge(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR, return_scale=False):
    """Scale the shorter edge to the given size, with a limit of `max_size` on
    the longer edge. If `max_size` is reached, then downscale so that the
    longer edge does not exceed max_size. only resize input image to target
    size and return scale.

    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        if return_scale:
            return im, im_scale
        else:
            return im
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        if return_scale:
            return padded_im, im_scale
        else:
            return padded_im



def get_bg_image(filename, imH, imW, channel=3):
        """keep aspect ratio of bg during resize target image size:

        imHximWxchannel.
        """
        target_size = min(imH, imW)
        max_size = max(imH, imW)
        real_hw_ratio = float(imH) / float(imW)
        bg_image = cv2.imread(filename)
        bg_h, bg_w, bg_c = bg_image.shape
        bg_image_resize = np.zeros((imH, imW, channel), dtype="uint8")
        if (float(imH) / float(imW) < 1 and float(bg_h) / float(bg_w) < 1) or (
            float(imH) / float(imW) >= 1 and float(bg_h) / float(bg_w) >= 1
        ):
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                if bg_h_new < bg_h:
                    bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
                else:
                    bg_image_crop = bg_image
            else:
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                if bg_w_new < bg_w:
                    bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
                else:
                    bg_image_crop = bg_image
        else:
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
            else:  # bg_h < bg_w
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                # logger.info(bg_w_new)
                bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
        bg_image_resize_0 = resize_short_edge(bg_image_crop, target_size, max_size)
        h, w, c = bg_image_resize_0.shape
        bg_image_resize[0:h, 0:w, :] = bg_image_resize_0
        return bg_image_resize


def replace_bg(im, im_mask, bg_filenames, truncate_fg = False):
        H, W = im.shape[:2]
        
        ind = random.randint(0, len(bg_filenames) - 1)
        filename = bg_filenames[ind]

        bg_img = get_bg_image(filename, H, W)

        mask = im_mask.copy()
        mask = mask.astype(np.bool)
        if truncate_fg:
            nonzeros = np.nonzero(mask.astype(np.uint8))
            x1, y1 = np.min(nonzeros, axis=1)
            x2, y2 = np.max(nonzeros, axis=1)
            c_h = 0.5 * (x1 + x2)
            c_w = 0.5 * (y1 + y2)
            rnd = random.random()
            # print(x1, x2, y1, y2, c_h, c_w, rnd, mask.shape)
            if rnd < 0.2:  # block upper
                c_h_ = int(random.uniform(x1, c_h))
                mask[:c_h_, :] = False
            elif rnd < 0.4:  # block bottom
                c_h_ = int(random.uniform(c_h, x2))
                mask[c_h_:, :] = False
            elif rnd < 0.6:  # block left
                c_w_ = int(random.uniform(y1, c_w))
                mask[:, :c_w_] = False
            elif rnd < 0.8:  # block right
                c_w_ = int(random.uniform(c_w, y2))
                mask[:, c_w_:] = False
            else:
                pass
        mask_bg = ~mask
        im[mask_bg] = bg_img[mask_bg]
        im = im.astype(np.uint8)
        return im, mask 


def get_background_fns(path):
    fns = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return fns


def build_augmentations(use_peper_salt, use_motion_blur):
    augmentations = []
    if use_peper_salt:
        augmentations.append(iaa.Sometimes(0.3, iaa.SaltAndPepper(0.05)))
    if use_motion_blur:
        augmentations.append(iaa.Sometimes(0.2, iaa.MotionBlur(k=5)))

    augmentations = augmentations + [iaa.Sometimes(0.4, iaa.CoarseDropout( p=0.1, size_percent=0.05) ),
                                    iaa.Sometimes(0.5, iaa.GaussianBlur(np.random.rand())),
                                    iaa.Sometimes(0.5, iaa.Add((-20, 20), per_channel=0.3)),
                                    iaa.Sometimes(0.4, iaa.Invert(0.20, per_channel=True)),
                                    iaa.Sometimes(0.5, iaa.Multiply((0.7, 1.4), per_channel=0.8)),
                                    iaa.Sometimes(0.5, iaa.Multiply((0.7, 1.4))),
                                    iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.0), per_channel=0.3))
                                    ]
  
    image_augmentations=iaa.Sequential(augmentations, random_order = False)
    return image_augmentations
        
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans
