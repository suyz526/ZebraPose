import os
import sys

sys.path.insert(0, os.getcwd())

from config_parser import parse_cfg
import argparse
from tqdm import tqdm

from tools_for_BOP import  bop_io

import torch
import numpy as np

from binary_code_helper.CNN_output_to_pose import load_dict_class_id_3D_points, CNN_outputs_to_object_pose
sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout

from model.BinaryCodeNet import BinaryCodeNet_Deeplab

from get_detection_results import get_detection_results_vivo
from common_ops import from_output_to_class_mask, from_output_to_class_binary_code, compute_original_mask
from tools_for_BOP.common_dataset_info import get_obj_info

import cv2

import json
from tools_for_BOP import write_to_cvs 

import torchvision.transforms as transforms
from PIL import Image

from bop_dataset_pytorch import get_final_Bbox, padding_Bbox, get_roi




def main(configs):
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    dataset_name = configs['dataset_name']
    training_data_folder=configs['training_data_folder']
    training_data_folder_2=configs['training_data_folder_2']
    test_folder=configs['test_folder']                                  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']              # the percentage of second dataset in the batch
    num_workers = configs['num_workers']
    train_obj_visible_theshold = configs['train_obj_visible_theshold']  # for test is always 0.1, for training we can set different values, usually 0.2
    #### network settings
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']        # network output size
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']              # now only support "L1" or "BCE"
    output_kernel_size = configs['output_kernel_size']                  # last layer kernel size
    resnet_layer = configs['resnet_layer']                              # usually resnet 34
    concat=configs['concat_encoder_decoder']      
    if 'efficientnet_key' in configs.keys():
        efficientnet_key = configs['efficientnet_key']             
    
    resize_method = configs['resize_method']
    Detection_reaults=configs['Detection_reaults']                       # for the test, the detected bounding box provided by GDR Net

    divide_number_each_itration = configs['divide_number_each_itration']
    number_of_itration = configs['number_of_itration']

    torch.manual_seed(0)     
    np.random.seed(0)      

    # get dataset informations
    dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global, cam_params = bop_io.get_dataset(bop_path, dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
    obj_name_obj_id, _ = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1) # now the obj_id started from 0
    
    mesh_path = model_plys[obj_id+1] # mesh_path is a dict, the obj_id should start from 1
    obj_diameter = model_info[str(obj_id+1)]['diameter']
    print("obj_diameter", obj_diameter)
    path_dict = os.path.join(dataset_dir, "models_GT_color", "Class_CorresPoint{:06d}.txt".format(obj_id+1))
    total_numer_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    divide_number_each_itration = int(divide_number_each_itration)
    total_numer_class = int(total_numer_class)
    number_of_itration = int(number_of_itration)
    if divide_number_each_itration ** number_of_itration != total_numer_class:
        raise AssertionError("the combination is not valid")
    GT_code_infos = [divide_number_each_itration, number_of_itration, total_numer_class]

    vertices = inout.load_ply(mesh_path)["pts"]

    # define test data loader
    if not bop_challange:
        dataset_dir_test, bop_test_folder,_,_,_,test_rgb_files,test_depth_files,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_dataset(bop_path, dataset_name, train=False, data_folder=test_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
    else:
        print("use BOP test images")
        dataset_dir_test, bop_test_folder,_,_,_,test_rgb_files,test_depth_files,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_bop_challange_test_data(bop_path, dataset_name, target_obj_id=obj_id+1, data_folder=test_folder)


    binary_code_length = number_of_itration
    print("predicted binary_code_length", binary_code_length)
    configs['binary_code_length'] = binary_code_length
   
    net = BinaryCodeNet_Deeplab(
                num_resnet_layers=resnet_layer, 
                concat=concat, 
                binary_code_length=binary_code_length, 
                divided_number_each_iteration = divide_number_each_itration, 
                output_kernel_size = output_kernel_size,
                efficientnet_key = efficientnet_key
            )

    if torch.cuda.is_available():
        net=net.cuda()

    checkpoint = torch.load( configs['checkpoint_file'] )
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()
    
    img_ids = []
    scene_ids = []
    estimated_Rs = []
    estimated_Ts = []
    scores = []

    if configs['use_icp']:
        #init the ICP Refiner
        from icp_module.ICP_Cosypose import ICPRefiner, read_depth
        test_img_example = cv2.imread(test_rgb_files[obj_id][0])
        print('example:', test_rgb_files[obj_id][0])
        icp_refiner = ICPRefiner(mesh_path, test_img_example.shape[1], test_img_example.shape[0], num_iters=100)
    
    test_rgb_files_no_duplicate = list(dict.fromkeys(test_rgb_files[obj_id]))

    if configs['detector'] == 'FCOS':
        from get_detection_results import get_detection_results_vivo
    elif configs['detector'] == 'MASKRCNN':
        from get_mask_rcnn_results import get_detection_results_vivo
    Bboxes = get_detection_results_vivo(Detection_reaults, test_rgb_files_no_duplicate, obj_id+1, 0)

    ##get camera parameters
    camera_params_dict = dict()
    for scene_id in os.listdir(bop_test_folder):
        current_dir = bop_test_folder+"/"+scene_id
        scene_params = inout.load_scene_camera(os.path.join(current_dir,"scene_camera.json"))     
        camera_params_dict[scene_id] = scene_params

    composed_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    for rgb_fn, Bboxes_frame in tqdm(Bboxes.items()):
        for Detected_Bbox in Bboxes_frame:
            rgb_fn_splitted = rgb_fn.split("/")
            scene_id = rgb_fn_splitted[-3]
            img_id = rgb_fn_splitted[-1].split(".")[0]
            rgb_fname = rgb_fn
           
            #Cam_K = camera_params_test[obj_id][0]['cam_K'].reshape((3,3))
            Cam_K = camera_params_dict[scene_id][int(img_id)]['cam_K'].reshape((3,3))
                        
            Bbox = Detected_Bbox['bbox_est']
            score = Detected_Bbox['score']
            rgb_img = cv2.imread(rgb_fname)
            Bbox = padding_Bbox(Bbox, 1.5)
            rgb_roi = get_roi(rgb_img, Bbox, 256, interpolation=cv2.INTER_LINEAR, resize_method=resize_method)
            #cv2.imshow("rgb_roi", rgb_roi)
            
            Bbox = get_final_Bbox(Bbox, resize_method, rgb_img.shape[1], rgb_img.shape[0])

            roi_pil = Image.fromarray(np.uint8(rgb_roi)).convert('RGB')
            input_x = composed_transforms_img(roi_pil)

            input_x=torch.unsqueeze(input_x, 0).cuda()
            pred_mask_prob, pred_code_prob = net(input_x)
            pred_masks = from_output_to_class_mask(pred_mask_prob)

            pred_codes = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, divided_num_each_interation=divide_number_each_itration, binary_code_length=binary_code_length)
           
            pred_codes = pred_codes.transpose(0, 2, 3, 1)

            pred_masks = pred_masks.transpose(0, 2, 3, 1)
            pred_masks = pred_masks.squeeze(axis=-1).astype('uint8')

            R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[0], pred_codes[0],
                                                                        Bbox, BoundingBox_CropSize_GT, divide_number_each_itration, dict_class_id_3D_points, 
                                                                        intrinsic_matrix=Cam_K)

            if success:     
                img_ids.append(img_id)
                scene_ids.append(scene_id)
                if configs['use_icp']:
                    if dataset_name != 'itodd':
                        depth_image = read_depth(os.path.join(dataset_dir_test, test_folder, scene_id, 'depth', "{:06d}.png".format(int(img_id))))
                    else:
                        depth_image = read_depth(os.path.join(dataset_dir_test, test_folder, scene_id, 'depth', "{:06d}.tif".format(int(img_id))))
                    if dataset_name == 'ycbv' or dataset_name == 'tless':
                        depth_image = depth_image * 0.1
                    full_mask = compute_original_mask(Bbox, test_img_example.shape[0], test_img_example.shape[1], pred_masks[0])
                    R_refined, t_refined = icp_refiner.refine_poses(t_predict*0.1, R_predict, full_mask, depth_image, Cam_K)
                    R_predict = R_refined
                    t_predict = t_refined*10.
                    t_predict = t_predict.reshape((3,1))    
                estimated_Rs.append(R_predict)
                estimated_Ts.append(t_predict)
                scores.append(score)

    print(len(scene_ids))
    print(len(img_ids))
    print(len(estimated_Rs))
    print(len(estimated_Ts))
    print(len(scores))

    cvs_path = os.path.join(eval_output_path, 'pose_result_bop/')
    if not os.path.exists(cvs_path):
        os.makedirs(cvs_path)
    write_to_cvs.write_cvs(cvs_path, "{}_{}".format(dataset_name, obj_name), obj_id+1, scene_ids, img_ids, estimated_Rs, estimated_Ts, scores)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str) # config file
    parser.add_argument('--obj_name', type=str)
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--ignore_bit', type=str)  # use the full 16 bit binary code, or ignore last n-bits
    parser.add_argument('--eval_output_path', type=str)
    parser.add_argument('--use_icp', type=str, choices=('True','False'), default='False') # config file
    args = parser.parse_args()
    config_file = args.cfg
    checkpoint_file = args.ckpt_file
    eval_output_path = args.eval_output_path
    obj_name = args.obj_name

    configs = parse_cfg(config_file)

    configs['obj_name'] = obj_name
    configs['use_icp'] = (args.use_icp == 'True')

    if configs['Detection_reaults'] != 'none':
        Detection_reaults = configs['Detection_reaults']
        dirname = os.path.dirname(__file__)
        Detection_reaults = os.path.join(dirname, Detection_reaults)
        configs['Detection_reaults'] = Detection_reaults

    configs['checkpoint_file'] = checkpoint_file
    configs['eval_output_path'] = eval_output_path

    configs['ignore_bit'] = int(args.ignore_bit)

    #print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)

    main(configs)