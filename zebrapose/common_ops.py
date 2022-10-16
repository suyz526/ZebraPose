import numpy as np
import torch
import cv2


def from_output_to_class_mask(pred_mask_prob, thershold=0.5):
    activation_function = torch.nn.Sigmoid()
    pred_mask_prob = activation_function(pred_mask_prob)
    pred_mask_prob = pred_mask_prob.detach().cpu().numpy()
    pred_mask = np.zeros(pred_mask_prob.shape)
    pred_mask[pred_mask_prob>thershold] = 1.
    return pred_mask

def from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, thershold=0.5, divided_num_each_interation=2, binary_code_length=16):
    if BinaryCode_Loss_Type == "BCE" or BinaryCode_Loss_Type == "L1":   
        activation_function = torch.nn.Sigmoid()
        pred_code_prob = activation_function(pred_code_prob)
        pred_code_prob = pred_code_prob.detach().cpu().numpy()
        pred_code = np.zeros(pred_code_prob.shape)
        pred_code[pred_code_prob>thershold] = 1.

    elif BinaryCode_Loss_Type == "CE":   
        activation_function = torch.nn.Softmax(dim=1)
        pred_code_prob = pred_code_prob.reshape(-1, divided_num_each_interation, pred_code_prob.shape[2], pred_code_prob.shape[3])
        pred_code_prob = activation_function(pred_code_prob)
        pred_code_prob = pred_code_prob.detach().cpu().numpy()
        pred_code = np.argmax(pred_code_prob, axis=1)
        pred_code = np.expand_dims(pred_code, axis=1)
        pred_code = pred_code.reshape(-1, binary_code_length, pred_code.shape[2], pred_code.shape[3])
        pred_code_prob = pred_code_prob.max(axis=1, keepdims=True)
        pred_code_prob = pred_code_prob.reshape(-1, binary_code_length, pred_code_prob.shape[2], pred_code_prob.shape[3])

    return pred_code

def compute_original_mask(Bbox, im_h, im_w, mask):
    if Bbox[0] < 0:
        x_offset = -1 * Bbox[0]
    else:
        x_offset = 0

    if Bbox[1] < 0:
        y_offset = -1 * Bbox[1]
    else:
        y_offset = 0
    
    resize_shape = max(Bbox[2], Bbox[3])
    full_mask_img = np.zeros((im_h, im_w), dtype=np.uint8)
    max_x = min(im_w,Bbox[0]+resize_shape)
    max_y = min(im_h,Bbox[1]+resize_shape)
    resize_x = min(im_w - Bbox[0],resize_shape)
    resize_y = min(im_h - Bbox[1],resize_shape)
    
    resized_mask = cv2.resize(mask, (resize_shape,resize_shape), interpolation=cv2.INTER_NEAREST)
    full_mask_img[Bbox[1]+y_offset:max_y,Bbox[0]+x_offset:max_x] = resized_mask[0+y_offset:resize_y,0+x_offset:resize_x]
    return full_mask_img

def get_batch_size(second_dataset_ratio, batch_size):
    batch_size_2_dataset = int(batch_size * second_dataset_ratio) 
    batch_size_1_dataset = batch_size - batch_size_2_dataset
    return batch_size_1_dataset, batch_size_2_dataset