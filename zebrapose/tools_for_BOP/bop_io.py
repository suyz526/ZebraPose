# modified from Pix2Pose https://github.com/kirumang/Pix2Pose

import numpy as np
import json
import os,sys
sys.path.append("..")
sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout


def get_target_list(target_path):      # get the test list for the bop test
    targets = inout.load_json(target_path)
    target_list=[]
    for i in range(len(targets)):
        tgt = targets[i]    
        im_id = tgt['im_id']
        inst_count = tgt['inst_count']
        obj_id = tgt['obj_id']
        scene_id = tgt['scene_id']
        
        target_list.append([scene_id,im_id,obj_id,inst_count])
       
    return target_list


def get_bop_challange_test_data(bop_dir, dataset, target_obj_id, data_folder='test'):
    print("get_bop_challange_test_data")
    bop_dataset_dir = os.path.join(bop_dir, dataset)

    model_dir = bop_dataset_dir+"/models_eval"
    model_info = inout.load_json(os.path.join(model_dir,"models_info.json"))
    model_ids = []
    for model_id in model_info.keys():
        ply_fn = os.path.join(model_dir,"obj_{:06d}.ply".format(int(model_id)))
        if(os.path.exists(ply_fn)): 
            model_ids.append(int(model_id)) #add model id only if the model.ply file exists
    model_ids = np.sort(np.array(model_ids))

    target_list_path = os.path.join(bop_dataset_dir, "test_targets_bop19.json")
    target_list = get_target_list(target_list_path)

    rgb_files_per_obj = [[] for x in range(model_ids.max())]
    depth_files_per_obj = [[] for x in range(model_ids.max())]
    mask_files_per_obj = [[] for x in range(model_ids.max())]
    mask_visib_files_per_obj = [[] for x in range(model_ids.max())]
    gts_per_obj = [[] for x in range(model_ids.max())]
    gt_infos_per_obj = [[] for x in range(model_ids.max())]
    params_per_obj = [[] for x in range(model_ids.max())]

    current_scene_id = -1
    for scene_id, im_id, obj_id, inst_count in target_list:
        if obj_id != target_obj_id:
            continue
        
        if current_scene_id != scene_id:
            scene_params = inout.load_scene_camera(os.path.join(bop_dataset_dir,data_folder, "{:06d}".format(scene_id),"scene_camera.json"))            
            scene_gts = inout.load_scene_gt(os.path.join(bop_dataset_dir,data_folder, "{:06d}".format(scene_id),"scene_gt.json"))
            scene_gt_infos = inout.load_scene_gt(os.path.join(bop_dataset_dir,data_folder, "{:06d}".format(scene_id),"scene_gt_info.json"))
            current_scene_id = scene_id

        rgb_files_per_obj[target_obj_id-1].append(os.path.join(bop_dataset_dir, data_folder, "{:06d}".format(scene_id), "rgb", "{:06d}.png".format(im_id)))
        depth_files_per_obj[target_obj_id-1].append(os.path.join(bop_dataset_dir, data_folder, "{:06d}".format(scene_id), "depth", "{:06d}.png".format(im_id)))
        
        gts = scene_gts[im_id]
        for counter, gt in enumerate(gts):
            if int(gt['obj_id']) == target_obj_id:
                mask_fn = os.path.join(bop_dataset_dir, data_folder,"{:06d}".format(scene_id), "mask","{:06d}_{:06d}.png".format(im_id, counter))   
                mask_visib_fn = os.path.join(bop_dataset_dir, data_folder,"{:06d}".format(scene_id), "mask_visib","{:06d}_{:06d}.png".format(im_id, counter))
                
                mask_files_per_obj[target_obj_id-1].append([mask_fn])
                mask_visib_files_per_obj[target_obj_id-1].append([mask_visib_fn])
                gts_per_obj[target_obj_id-1].append(gt)
                gt_infos_per_obj[target_obj_id-1].append(scene_gt_infos[im_id][counter])  
                params_per_obj[target_obj_id-1].append(scene_params[im_id])
                break
    return bop_dataset_dir,[],[],[],[],rgb_files_per_obj,depth_files_per_obj,mask_files_per_obj,mask_visib_files_per_obj,gts_per_obj,gt_infos_per_obj,[],params_per_obj


def get_dataset(bop_dir,dataset,train=True,incl_param=False ,eval_model=False, data_folder='None', data_per_obj=False, train_obj_visible_theshold=0.1):
    """
    bop_dir:            dir to bop folder
    dataset:            dataset name in bop_dir, like lm, lmo, tudl
    train:              if use this function for training data
    incl_param:         if return the camera parameter for each image
    eval_model:         if use the funtion to get eval mesh? The meshsize is smaller, so it is faster for the evaluation
    data_folder:        e.g. train_real, train_pbr
    data_per_obj:       if the ground truth data for an image only contains the ground truth for one obj?
                        This leads to diffent type of output data of this function.
                        if False, rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,params are a list
                        if True, rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,params are a list of list. The first index is obj_id
    """
    #return serialized datset information
    if eval_model:
        postfix_model = '_eval'
    else:
        postfix_model = ''

    bop_dataset_dir = os.path.join(bop_dir,dataset)
    target_dir = os.path.join(bop_dataset_dir,data_folder)
    model_dir = bop_dataset_dir+"/models"+postfix_model
    
    model_info = inout.load_json(os.path.join(model_dir,"models_info.json"))
    if(dataset=='ycbv'):
        cam_param_global = inout.load_cam_params(os.path.join(bop_dataset_dir,"camera_uw.json"))
    elif(dataset=='tless' or dataset=='hb'):
        cam_param_global = inout.load_cam_params(os.path.join(bop_dataset_dir,"camera_primesense.json"))
    else:
        cam_param_global = inout.load_cam_params(os.path.join(bop_dataset_dir,"camera.json"))
    
    im_size=np.array(cam_param_global['im_size'])[::-1]
    
        
    model_plys={}
    model_ids = []
    for model_id in model_info.keys():
        ply_fn = os.path.join(model_dir,"obj_{:06d}.ply".format(int(model_id)))
        if(os.path.exists(ply_fn)): 
            model_ids.append(int(model_id)) #add model id only if the model.ply file exists

    model_ids = np.sort(np.array(model_ids))
    for model_id in model_ids:
        ply_fn = os.path.join(model_dir,"obj_{:06d}.ply".format(int(model_id)))
        model_plys[int(model_id)] = ply_fn
        print(ply_fn)

    print("if models are not fully listed above, please make sure there are ply files available")

    rgb_files_dataset = []
    depth_files_dataset = []
    mask_files_dataset = []
    mask_visib_files_dataset = []
    gts_dataset = []
    gt_infos_dataset = []
    params_dataset = []

    max_id = model_ids.max()
    if dataset == 'lmo':
        max_id = 15
    rgb_files_per_obj = [[] for x in range(max_id)]
    depth_files_per_obj = [[] for x in range(max_id)]
    mask_files_per_obj = [[] for x in range(max_id)]
    mask_visib_files_per_obj = [[] for x in range(max_id)]
    gts_per_obj = [[] for x in range(max_id)]
    gt_infos_per_obj = [[] for x in range(max_id)]
    params_per_obj = [[] for x in range(max_id)]

    if(os.path.exists(target_dir)):        
        for dir in os.listdir(target_dir): #loop over a seqeunce 
            current_dir = target_dir+"/"+dir
            if os.path.exists(os.path.join(current_dir,"scene_camera.json")):
                scene_params = inout.load_scene_camera(os.path.join(current_dir,"scene_camera.json"))            
                scene_gt_fn = os.path.join(current_dir,"scene_gt.json")  # fn for filename
                scene_gt_info_fn = os.path.join(current_dir,"scene_gt_info.json")
                has_gt=False
                if os.path.exists(scene_gt_fn) and os.path.exists(scene_gt_info_fn):
                    scene_gts = inout.load_scene_gt(scene_gt_fn)
                    scene_gt_infos = inout.load_scene_gt(scene_gt_info_fn)
                    has_gt=True
    
                for img_id in sorted(scene_params.keys()):
                    im_id = int(img_id)
                    if(dataset=="itodd" and not(train)):
                        rgb_fn = os.path.join(current_dir+"/gray","{:06d}.tif".format(im_id))
                    else:
                        rgb_fn = os.path.join(current_dir+"/rgb","{:06d}.png".format(im_id))
                    depth_fn = os.path.join(current_dir+"/depth","{:06d}.png".format(im_id))

                    if not os.path.exists(rgb_fn):
                        rgb_fn_no_surfix = rgb_fn[:-4]
                        rgb_fn = rgb_fn_no_surfix + ".jpg"
                            
                    if data_per_obj:
                        visib_thershold = 0.1
                        if train:
                            visib_thershold = train_obj_visible_theshold
                        gts = scene_gts[im_id]
                        for counter, gt in enumerate(gts):
                            visib_fract = scene_gt_infos[im_id][counter]['visib_fract']
                            if visib_fract > visib_thershold:
                                obj_id = int(gt['obj_id']-1)
                                mask_fn = os.path.join(current_dir+"/mask","{:06d}_{:06d}.png".format(im_id, counter))
                                
                                mask_visib_fn = os.path.join(current_dir+"/mask_visib","{:06d}_{:06d}.png".format(im_id, counter))
                                
                                rgb_files_per_obj[obj_id].append(rgb_fn)
                                depth_files_per_obj[obj_id].append(depth_fn) 
                                mask_files_per_obj[obj_id].append([mask_fn])
                                mask_visib_files_per_obj[obj_id].append([mask_visib_fn])
                                gts_per_obj[obj_id].append(gt)
                                gt_infos_per_obj[obj_id].append(scene_gt_infos[im_id][counter])  
                                params_per_obj[obj_id].append(scene_params[im_id])  
                    else:
                        rgb_files_dataset.append(rgb_fn)
                        depth_files_dataset.append(depth_fn)
                        if(has_gt):
                            gts_dataset.append(scene_gts[im_id])
                            gt_infos_dataset.append(scene_gt_infos[im_id])
                        params_dataset.append(scene_params[im_id])  
                        mask_fns = []
                        mask_visib_fns = []
                        for counter, gt in enumerate(scene_gts[im_id]):
                            mask_fn = os.path.join(current_dir+"/mask","{:06d}_{:06d}.png".format(im_id, counter))
                            #if not os.path.exists(mask_fn):
                            #    print(mask_fn, " not exist!!!")
                            mask_visib_fn = os.path.join(current_dir+"/mask_visib","{:06d}_{:06d}.png".format(im_id, counter))
                            #if not os.path.exists(mask_visib_fn):
                            #    print(mask_visib_fn, " not exist!!!")
                            mask_fns.append(mask_fn)
                            mask_visib_fns.append(mask_visib_fn)
                        mask_files_dataset.append(mask_fns)
                        mask_visib_files_dataset.append(mask_visib_fns)
                            
    if data_per_obj:
        rgb_files = rgb_files_per_obj
        depth_files = depth_files_per_obj
        mask_files = mask_files_per_obj
        mask_visib_files = mask_visib_files_per_obj
        gts = gts_per_obj
        gt_infos = gt_infos_per_obj
        params = params_per_obj
    else:
        rgb_files = rgb_files_dataset
        depth_files = depth_files_dataset
        mask_files = mask_files_dataset
        mask_visib_files = mask_files_dataset
        gts = gts_dataset
        gt_infos = gt_infos_dataset
        params = params_dataset

    if(incl_param):
        return bop_dataset_dir,target_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global,params
    else:
        return bop_dataset_dir,target_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global


def get_dataset_basic_info(bop_dir, dataset, train=True):
    #return serialized datset information
    if not train:
        postfix_model = '_eval'
    else:
        postfix_model = ''

    bop_dataset_dir = os.path.join(bop_dir,dataset)
    model_dir = bop_dataset_dir+"/models"+postfix_model
    
    model_info = inout.load_json(os.path.join(model_dir,"models_info.json"))

    model_plys={}
    model_ids = []
    for model_id in model_info.keys():
        ply_fn = os.path.join(model_dir,"obj_{:06d}.ply".format(int(model_id)))
        if(os.path.exists(ply_fn)): 
            model_ids.append(int(model_id)) #add model id only if the model.ply file exists

    model_ids = np.sort(np.array(model_ids))
    for model_id in model_ids:
        ply_fn = os.path.join(model_dir,"obj_{:06d}.ply".format(int(model_id)))
        model_plys[int(model_id)] = ply_fn
        print(ply_fn)

    print("if models are not fully listed above, please make sure there are ply files available")

    return bop_dataset_dir, model_plys,model_info
