import os
import sys
import argparse

sys.path.append("../zebrapose/tools_for_BOP")
import bop_io

import cv2
import Render
import numpy as np

from tqdm import tqdm

def generate_GT_images(bop_path, dataset_name, force_rewrite, is_training_data, data_folder, start_obj_id, end_obj_id):
    dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global,scene_cam = bop_io.get_dataset(bop_path,dataset_name,train=is_training_data, incl_param=True, data_folder=data_folder)

    target_dir = os.path.join(dataset_dir,data_folder + '_GT')

    im_width,im_height =cam_param_global['im_size'] 
    if dataset_name == 'tless':
        im_width = 720
        im_height = 540
        if data_folder == 'train_primesense':
            im_width = 400
            im_height = 400

    cam_K = cam_param_global['K']
    camera_parameters = np.array([im_width, im_height, cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2]])
    print(camera_parameters)
    Render.init(camera_parameters, 1)
    model_scale = 0.1
    
    for model_to_render in range(start_obj_id, end_obj_id):
        # only bind 1 model each time
        model_to_render = int(model_to_render)
        ply_fn = dataset_dir+"/models_GT_color/obj_{:06d}.ply".format(int(model_ids[model_to_render]))
        print("bind ", ply_fn, " to the render buffer position", 0)
        Render.bind_3D_model(ply_fn, 0, model_scale)

        print("rgb_files:", len(rgb_files))
        print("gts:", len(gts))
        print("scene_cam:", len(scene_cam))
        
        ##for each image render the ground truth 
        for img_id in tqdm(range(len(rgb_files))):
            rgb_path = rgb_files[img_id]
            rgb_path = rgb_path.split("/")
            scene_id = rgb_path[-3]
            image_name = rgb_path[-1][:-4]

            GT_img_dir = os.path.join(target_dir, scene_id)

            if not(os.path.exists(GT_img_dir)):
                os.makedirs(GT_img_dir)

            cam_K_local = np.array(scene_cam[img_id]["cam_K"]).reshape(3,3)
            camera_parameters_local = np.array([im_width, im_height, cam_K_local[0,0], cam_K_local[1,1], cam_K_local[0,2], cam_K_local[1,2]])
            Render.reset_camera_params(camera_parameters_local)
            
            #visible side
            for count, gt in enumerate(gts[img_id]):
                GT_img_fn = os.path.join(GT_img_dir,"{}_{:06d}.png".format(image_name,count))         
               
                obj_id_render = int(gt['obj_id']-1)
                if obj_id_render != model_to_render:
                    continue        
        
                if(os.path.exists(GT_img_fn) and not force_rewrite):
                    continue
                else:
                    tra_pose = np.array(gt['cam_t_m2c'])
                    tra_pose = tra_pose * model_scale
                    rot_pose = np.array(gt['cam_R_m2c']).flatten()

                    Render.render_GT_visible_side(tra_pose, rot_pose, 0, GT_img_fn)                  

    Render.delete()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate image labels for bop dataset')
    parser.add_argument('--bop_path', help='path to the bop folder', required=True)
    parser.add_argument('--dataset_name', help='the folder name of the dataset in the bop folder', required=True)
    parser.add_argument('--force_rewrite', choices=['True', 'False'], default='False', help='if rewrite the exist data', required=True)
    parser.add_argument('--is_training_data', choices=['True', 'False'], default='True', help='if is applied to training data ', required=True)
    parser.add_argument('--data_folder', help='which training data')
    parser.add_argument('--start_obj_id', help='start_obj_id')
    parser.add_argument('--end_obj_id', help='which training data')

    args = parser.parse_args()

    bop_path = args.bop_path
    dataset_name = args.dataset_name
    force_rewrite = args.force_rewrite == 'True'
    is_training_data = args.is_training_data == 'True'
    data_folder = args.data_folder
    start_obj_id = int(args.start_obj_id)
    end_obj_id = int(args.end_obj_id)
    
    generate_GT_images(bop_path, dataset_name, force_rewrite, is_training_data, data_folder, start_obj_id, end_obj_id)