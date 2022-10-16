import os

import torch
import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset

import sys
from binary_code_helper.class_id_encoder_decoder import RGB_image_to_class_id_image, class_id_image_to_class_code_images
import torchvision.transforms as transforms


import GDR_Net_Augmentation
from GDR_Net_Augmentation import get_affine_transform

def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img

def crop_square_resize(img, Bbox, crop_size=None, interpolation=None):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh

    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
    if bh > bw:
        x1 = bbox_center[0] - bh/2
        x2 = bbox_center[0] + bh/2
    else:
        y1 = bbox_center[1] - bw/2
        y2 = bbox_center[1] + bw/2

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if img.ndim > 2:
        roi_img = np.zeros((max(bh, bw), max(bh, bw), img.shape[2]), dtype=img.dtype)
    else:
        roi_img = np.zeros((max(bh, bw), max(bh, bw)), dtype=img.dtype)
    roi_x1 = max((0-x1), 0)
    x1 = max(x1, 0)
    roi_x2 = roi_x1 + min((img.shape[1]-x1), (x2-x1))
    roi_y1 = max((0-y1), 0)
    y1 = max(y1, 0)
    roi_y2 = roi_y1 + min((img.shape[0]-y1), (y2-y1))
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])

    roi_img[roi_y1:roi_y2, roi_x1:roi_x2] = img[y1:y2, x1:x2].copy()
    roi_img = cv2.resize(roi_img, (crop_size,crop_size), interpolation=interpolation)
    return roi_img

def crop_resize(img, Bbox, crop_size=None, interpolation=None):
    x1 = max(0, Bbox[0])
    x2 = min(img.shape[1], Bbox[0]+Bbox[2])
    y1 = max(0, Bbox[1])
    y2 = min(img.shape[0], Bbox[1]+Bbox[3])
    ####
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])
    ####

    img = img[y1:y2, x1:x2]
    roi_img = cv2.resize(img, (crop_size, crop_size), interpolation = interpolation)
    return roi_img

def get_scale_and_Bbox_center(Bbox, image):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh

    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
    if bh > bw:
        x1 = bbox_center[0] - bh/2
        x2 = bbox_center[0] + bh/2
    else:
        y1 = bbox_center[1] - bw/2
        y2 = bbox_center[1] + bw/2

    scale = max(bh, bw)
    scale = min(scale, max(image.shape[0], image.shape[1])) *1.0
    return scale, bbox_center

def get_roi(input, Bbox, crop_size, interpolation, resize_method):
    if resize_method == "crop_resize":
        roi = crop_resize(input, Bbox, crop_size, interpolation = interpolation)
        return roi
    elif resize_method == "crop_resize_by_warp_affine":
        scale, bbox_center = get_scale_and_Bbox_center(Bbox, input)
        roi = crop_resize_by_warp_affine(input, bbox_center, scale, crop_size, interpolation = interpolation)
        return roi
    elif resize_method == "crop_square_resize":
        roi = crop_square_resize(input, Bbox, crop_size, interpolation=interpolation)
        return roi
    else:
        raise NotImplementedError(f"unknown decoder type: {resize_method}")

def padding_Bbox(Bbox, padding_ratio):
    x1 = Bbox[0]
    x2 = Bbox[0] + Bbox[2]
    y1 = Bbox[1]
    y2 = Bbox[1] + Bbox[3]

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1

    padded_bw = int(bw * padding_ratio)
    padded_bh = int(bh * padding_ratio)
        
    padded_Box = np.array([int(cx-padded_bw/2), int(cy-padded_bh/2), int(padded_bw), int(padded_bh)])
    return padded_Box

def aug_Bbox(GT_Bbox, padding_ratio):
    x1 = GT_Bbox[0].copy()
    x2 = GT_Bbox[0] + GT_Bbox[2]
    y1 = GT_Bbox[1].copy()
    y2 = GT_Bbox[1] + GT_Bbox[3]

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1

    scale_ratio = 1 + 0.25 * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
    shift_ratio = 0.25 * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
    bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
    # 1.5 is the additional pad scale
    augmented_bw = int(bw * scale_ratio * padding_ratio)
    augmented_bh = int(bh * scale_ratio * padding_ratio)
    
    augmented_Box = np.array([int(bbox_center[0]-augmented_bw/2), int(bbox_center[1]-augmented_bh/2), augmented_bw, augmented_bh])
    return augmented_Box

def get_final_Bbox(Bbox, resize_method, max_x, max_y):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh
    if resize_method == "crop_square_resize" or resize_method == "crop_resize_by_warp_affine":
        bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        if bh > bw:
            x1 = bbox_center[0] - bh/2
            x2 = bbox_center[0] + bh/2
        else:
            y1 = bbox_center[1] - bw/2
            y2 = bbox_center[1] + bw/2
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        Bbox = np.array([x1, y1, x2-x1, y2-y1])

    elif resize_method == "crop_resize":
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, max_x)
        y2 = min(y2, max_y)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        Bbox = np.array([x1, y1, x2-x1, y2-y1])

    return Bbox

class bop_dataset_single_obj_pytorch(Dataset):
    def __init__(self, dataset_dir, data_folder, rgb_files, mask_files, mask_visib_files, gts, gt_infos, cam_params, 
                        is_train, crop_size_img, crop_size_gt, GT_code_infos, padding_ratio=1.5, resize_method="crop_resize", 
                        use_peper_salt=False, use_motion_blur=False, Detect_Bbox=None):
        # gts: rotation and translation
        # gt_infos: bounding box
        self.rgb_files = rgb_files
        self.mask_visib_files = mask_visib_files
        self.mask_files = mask_files
        self.gts = gts
        self.gt_infos = gt_infos
        self.cam_params = cam_params
        self.dataset_dir = dataset_dir
        self.data_folder = data_folder
        self.is_train = is_train
        self.GT_code_infos = GT_code_infos
        self.crop_size_img = crop_size_img
        self.crop_size_gt = crop_size_gt
        self.resize_method = resize_method
        self.Detect_Bbox = Detect_Bbox
        self.padding_ratio = padding_ratio
        self.use_peper_salt = use_peper_salt
        self.use_motion_blur = use_motion_blur

        self.nSamples = len(self.rgb_files)
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        # return training image, mask, bounding box, R, T, GT_image
        rgb_fn = self.rgb_files[index]
        mask_visib_fns = self.mask_visib_files[index]
        mask_fns = self.mask_files[index]

        x = cv2.imread(rgb_fn)
        mask = cv2.imread(mask_visib_fns[0], 0)
        entire_mask = cv2.imread(mask_fns[0], 0)

        #rgb_files    ...datasetpath/train/scene_id/rgb/img.png
        rgb_fn = rgb_fn.split("/")
        scene_id = rgb_fn[-3]
        GT_image_name = mask_visib_fns[0].split("/")[-1]
        
        GT_img_dir = os.path.join(self.dataset_dir, self.data_folder + '_GT', scene_id)
        GT_img_fn = os.path.join(GT_img_dir, GT_image_name)        
        GT_img = cv2.imread(GT_img_fn)

        gt = self.gts[index]
        gt_info = self.gt_infos[index]

        if gt == None:  
            R = np.array(range(9)).reshape(3,3) 
            t = np.array(range(3)) 
            Bbox = np.array([1,1,1,1])
        else:
            R = np.array(gt['cam_R_m2c']).reshape(3,3) 
            t = np.array(gt['cam_t_m2c']) 
            Bbox = np.array(gt_info['bbox_visib'])

        cam_param = self.cam_params[index]['cam_K'].reshape((3,3))

        #print("show original train image")
        #self.visulize(x, entire_mask, mask, GT_img_visible, GT_img_invisible, Bbox)
        #print(Bbox)
        if self.is_train:           
            x = self.apply_augmentation(x)
            
            Bbox = aug_Bbox(Bbox, padding_ratio=self.padding_ratio)

            roi_x = get_roi(x, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR, resize_method = self.resize_method)
            roi_GT_img = get_roi(GT_img, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST, resize_method = self.resize_method)
            roi_mask = get_roi(mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST, resize_method = self.resize_method)
            roi_entire_mask = get_roi(entire_mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST, resize_method = self.resize_method)
            
            Bbox = get_final_Bbox(Bbox, self.resize_method, x.shape[1], x.shape[0])

            #print("show cropped train image")
            #self.visulize(roi_x, roi_entire_mask, roi_mask, roi_GT_img_visible, roi_GT_img_invisible, None)            
        else:   
            if self.Detect_Bbox!=None:
                # replace the Bbox with detected Bbox
                Bbox = self.Detect_Bbox[index]
                if Bbox == None: #no valid detection, give a dummy input
                    roi_x = torch.zeros((3, self.crop_size_img, self.crop_size_img))
                    roi_GT_img = torch.zeros((int(self.GT_code_infos[1]), int(self.crop_size_gt), int(self.crop_size_gt)))
                    roi_mask = torch.zeros((int(self.crop_size_gt), int(self.crop_size_gt)))
                    roi_entire_mask = torch.zeros((int(self.crop_size_gt), int(self.crop_size_gt)))
                    Bbox = np.array([0,0,0,0], dtype='int')
                    return roi_x, roi_entire_mask, roi_mask, R, t, Bbox, roi_GT_img, cam_param

            if not os.path.exists(GT_img_fn):
                # some test fold doesn't provide GT, fill GT with dummy value
                GT_img = np.zeros(x.shape)
                mask = np.zeros((x.shape[0], x.shape[1]))
                entire_mask = np.zeros((x.shape[0], x.shape[1]))

            Bbox = padding_Bbox(Bbox, padding_ratio=self.padding_ratio)
            roi_x = get_roi(x, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR, resize_method = self.resize_method)
            roi_GT_img = get_roi(GT_img, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST, resize_method = self.resize_method)
            roi_mask = get_roi(mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST, resize_method = self.resize_method)
            roi_entire_mask = get_roi(entire_mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST, resize_method = self.resize_method)
            
            #print("show test_image")
            Bbox = get_final_Bbox(Bbox, self.resize_method, x.shape[1], x.shape[0])
            # self.visulize(roi_x, roi_entire_mask, roi_mask, roi_GT_img_visible, roi_GT_img_invisible, None)

        class_id_image= RGB_image_to_class_id_image(roi_GT_img)
        roi_GT_img = class_id_image_to_class_code_images(class_id_image, self.GT_code_infos[0], self.GT_code_infos[1], self.GT_code_infos[2])

        # add the augmentations and transfrom in torch tensor
        roi_x, roi_entire_mask, roi_mask, class_code_images = self.transform_pre(roi_x, roi_entire_mask, roi_mask, roi_GT_img)
        # for single obj, only one gt
        return roi_x, roi_entire_mask, roi_mask, R, t, Bbox, class_code_images, cam_param

    def visulize(self, x, entire_mask, mask, GT_img_visible, GT_img_invisible, Bbox):
        cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('entire_mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('GT_img_visible', cv2.WINDOW_NORMAL)
        cv2.namedWindow('GT_img_invisible', cv2.WINDOW_NORMAL)

        x_ = x.copy()
        if Bbox is not None:
            cv2.rectangle(x_,(Bbox[0],Bbox[1]),(Bbox[0]+Bbox[2] ,Bbox[1]+Bbox[3] ),(0,255,0),3) 
        cv2.imshow('rgb',x_)
        cv2.imshow('mask',mask)
        cv2.imshow('entire_mask',entire_mask)
        
        cv2.imshow('GT_img_visible',GT_img_visible)
        cv2.imshow('GT_img_invisible',GT_img_invisible)

        cv2.waitKey(0)

    def transform_pre(self, sample_x,sample_entire_mask, sample_mask,gt_code):
        composed_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        x_pil = Image.fromarray(np.uint8(sample_x)).convert('RGB')

        sample_entire_mask = sample_entire_mask / 255.
        sample_entire_mask = torch.from_numpy(sample_entire_mask).type(torch.float)
        sample_mask = sample_mask / 255.
        sample_mask = torch.from_numpy(sample_mask).type(torch.float)
        gt_code = torch.from_numpy(gt_code).permute(2, 0, 1) 
    
        return composed_transforms_img(x_pil), sample_entire_mask, sample_mask, gt_code

    def apply_augmentation(self, x):
        augmentations = GDR_Net_Augmentation.build_augmentations(self.use_peper_salt, self.use_motion_blur)      
        color_aug_prob = 0.8
        if np.random.rand() < color_aug_prob:
            x = augmentations.augment_image(x)

        return x