""" sym aware labels
    Only generate labels for sym object defined in BOP datasets, the generated labels store in *_GT_v2.
    Usage: python generate_training_labels_for_BOP_v2 --bop_path Path/to/BOP_DATASETS --dataset_name tless --data_folder train_pbr 
"""

import os
import sys
import argparse

sys.path.append("../zebrapose/tools_for_BOP")
import bop_io

import cv2
import Render
import numpy as np
from tqdm import tqdm


def generate_GT_images(bop_path, dataset_name, force_rewrite, is_training_data, data_folder):
    dataset_dir, source_dir, model_plys, model_info, model_ids, rgb_files, depth_files, mask_files, mask_visib_files, gts, gt_infos, cam_param_global, scene_cam = bop_io.get_dataset(
        bop_path, dataset_name, train=is_training_data, incl_param=True, data_folder=data_folder)
    sym_obj_id = [int(i) for i in model_info.keys() if 'symmetries_discrete' in model_info[i].keys() or 'symmetries_continuous' in model_info[i].keys()]
    target_dir = os.path.join(dataset_dir, data_folder + '_GT_v2')

    im_width, im_height = cam_param_global['im_size']
    if dataset_name == 'tless':
        im_width = 720
        im_height = 540
        if data_folder == 'train_primesense':
            im_width = 400
            im_height = 400

    cam_K = cam_param_global['K']
    camera_parameters = np.array([im_width, im_height, cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]])
    print(camera_parameters)
    Render.init(camera_parameters, 1)
    model_scale = 0.1

    for model_to_render in sym_obj_id:
        # only bind 1 model each time
        assert model_to_render in model_ids
        ply_fn = dataset_dir + "/models_GT_color/obj_{:06d}.ply".format(model_to_render)
        print("bind ", ply_fn, " to the render buffer position", 0)
        Render.bind_3D_model(ply_fn, 0, model_scale)

        print("rgb_files:", len(rgb_files))
        print("gts:", len(gts))
        print("scene_cam:", len(scene_cam))

        ## for each image render the ground truth
        for img_id in tqdm(range(len(rgb_files))):
            rgb_path = rgb_files[img_id]
            rgb_path = rgb_path.split("/")
            scene_id = rgb_path[-3]
            image_name = rgb_path[-1][:-4]

            GT_img_dir = os.path.join(target_dir, scene_id)

            if not (os.path.exists(GT_img_dir)):
                os.makedirs(GT_img_dir)

            cam_K_local = np.array(scene_cam[img_id]["cam_K"]).reshape(3, 3)
            camera_parameters_local = np.array(
                [im_width, im_height, cam_K_local[0, 0], cam_K_local[1, 1], cam_K_local[0, 2], cam_K_local[1, 2]])
            Render.reset_camera_params(camera_parameters_local)

            # visible side
            for count, gt in enumerate(gts[img_id]):
                GT_img_fn = os.path.join(GT_img_dir, "{}_{:06d}.png".format(image_name, count))

                if gt['obj_id'] != model_to_render:
                    continue

                if os.path.exists(GT_img_fn) and not force_rewrite:
                    continue
                else:
                    tra_pose = np.array(gt['cam_t_m2c'])
                    rot_pose = np.array(gt['cam_R_m2c'])

                    if model_to_render in sym_obj_id:
                        rot_pose, tra_pose = modified_gt_for_symmetry(rot_pose, tra_pose, model_info[str(model_to_render)])

                    tra_pose = tra_pose * model_scale
                    rot_pose = rot_pose.flatten()
                    Render.render_GT_visible_side(tra_pose, rot_pose, 0, GT_img_fn)

    Render.delete()


def modified_gt_for_symmetry(rot_pose, tra_pose, model_info):
    """ modify gt to minimize |RS-I| """
    if 'symmetries_continuous' in model_info and 'symmetries_discrete' in model_info:
        ''' TODO this condition hasn't been tested yet '''
        #### step1: verify the case can be solved by the code
        assert len(model_info['symmetries_continuous']) == 1
        assert model_info['symmetries_continuous'][0]['axis'] == [0, 0, 1]
        assert model_info['symmetries_continuous'][0]['offset'] == [0, 0, 0]
        #### step2: store the symmetries_descrete in trans_discs
        trans_discs = [{'R': np.eye(3), 't': np.array([[0, 0, 0]]).T}]  # Identity.
        for sym in model_info['symmetries_discrete']:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_discs.append({'R': R, 't': t})
        ### step3: store the best possible poses for every descrete symmetries counterpart in pose_tmps
        ### (considering descrete symmetries and continuous symmetries)
        pose_tmps = []
        for trans_disc in trans_discs:
            tra_pose_tmp = rot_pose.dot(trans_disc['t']) + tra_pose
            rot_pose_tmp = rot_pose.dot(trans_disc['R'])
            R11 = rot_pose_tmp[0, 0]
            R12 = rot_pose_tmp[0, 1]
            R21 = rot_pose_tmp[1, 0]
            R22 = rot_pose_tmp[1, 1]
            theta = np.arctan((R12 - R21) / (R11 + R22))
            if not np.sin(theta) * (R21 - R12) < np.cos(theta) * (R11 + R22):
                theta = theta + np.pi
            S = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
            tra_pose_tmp = rot_pose_tmp.dot(np.array([[0.], [0.], [0.]])) + tra_pose_tmp
            rot_pose_tmp = rot_pose_tmp.dot(S)
            pose_tmps.append({'R': rot_pose_tmp, 't': tra_pose_tmp})
        ### step4: choose best from pose_tmps
        best_pose = None
        froebenius_norm = 1e8
        for pose_tmp in pose_tmps:
            tmp_froebenius_norm = np.linalg.norm(pose_tmp['R'] - np.eye(3))
            if tmp_froebenius_norm < froebenius_norm:
                froebenius_norm = tmp_froebenius_norm
                best_pose = pose_tmp
        tra_pose = best_pose['t']
        rot_pose = best_pose['R']

    elif 'symmetries_continuous' in model_info:
        # currently not support for both discrete and continuous symmetry
        assert 'symmetries_discrete' not in model_info
        # currently not support for multi symmetries continuous
        assert len(model_info['symmetries_continuous']) == 1
        sym = model_info['symmetries_continuous'][0]
        if sym['axis'] == [0, 0, 1] and sym['offset'] == [0, 0, 0]:
            R11 = rot_pose[0, 0]
            R12 = rot_pose[0, 1]
            R21 = rot_pose[1, 0]
            R22 = rot_pose[1, 1]
            theta = np.arctan((R12 - R21) / (R11 + R22))
            if not np.sin(theta) * (R21 - R12) < np.cos(theta) * (R11 + R22):
                theta = theta + np.pi
            S = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
            tra_pose = rot_pose.dot(np.array([[0.], [0.], [0.]])) + tra_pose
            rot_pose = rot_pose.dot(S)
        elif sym['axis'] == [0, 1, 0] and sym['offset'] == [0, 0, 0]:
            R11 = rot_pose[0, 0]
            R13 = rot_pose[0, 2]
            R31 = rot_pose[2, 0]
            R33 = rot_pose[2, 2]
            theta = np.arctan((R31 - R13) / (R11 + R33))
            if not np.sin(theta) * (R13 - R31) < np.cos(theta) * (R11 + R33):
                theta = theta + np.pi
            S = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
            tra_pose = rot_pose.dot(np.array([[0.], [0.], [0.]])) + tra_pose
            rot_pose = rot_pose.dot(S)
        elif sym['axis'] == [1, 0, 0] and sym['offset'] == [0, 0, 0]:
            R22 = rot_pose[1, 1]
            R23 = rot_pose[1, 2]
            R32 = rot_pose[2, 1]
            R33 = rot_pose[2, 2]
            theta = np.arctan((R32 - R23) / (R22 + R33))
            if not (R22 + R33) * np.cos(theta) + (R32 - R23) * np.sin(theta) > 0:
                theta = theta + np.pi
            S = np.array([[1, 0, 0],
                          [0, np.cos(theta), np.sin(theta)],
                          [0, -np.sin(theta), np.cos(theta)]])
            tra_pose = rot_pose.dot(np.array([[0.], [0.], [0.]])) + tra_pose
            rot_pose = rot_pose.dot(S)
        else:
            raise NotImplementedError
    elif 'symmetries_discrete' in model_info:
        # currently not support for both discrete and continuous symmetry
        assert 'symmetries_continuous' not in model_info
        trans_disc = [{'R': np.eye(3), 't': np.array([[0, 0, 0]]).T}]  # Identity.
        for sym in model_info['symmetries_discrete']:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_disc.append({'R': R, 't': t})
        best_R = None
        best_t = None
        froebenius_norm = 1e8
        for sym in trans_disc:
            R = sym['R']
            t = sym['t']
            tmp_froebenius_norm = np.linalg.norm(rot_pose.dot(R)-np.eye(3))
            if tmp_froebenius_norm < froebenius_norm:
                froebenius_norm = tmp_froebenius_norm
                best_R = R
                best_t = t
        tra_pose = rot_pose.dot(best_t) + tra_pose
        rot_pose = rot_pose.dot(best_R)
    elif (not 'symmetries_discrete' in model_info) and (not 'symmetries_continuous' in model_info):
        return rot_pose, tra_pose
    else:
        raise NotImplementedError
    return rot_pose, tra_pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate image labels for bop dataset')
    parser.add_argument('--bop_path', help='path to the bop folder', required=True)
    parser.add_argument('--dataset_name', help='the folder name of the dataset in the bop folder', required=True)
    parser.add_argument('--force_rewrite', choices=['True', 'False'], default='False', help='if rewrite the exist data',
                        required=True)
    parser.add_argument('--is_training_data', choices=['True', 'False'], default='True',
                        help='if is applied to training data ', required=True)
    parser.add_argument('--data_folder', help='which training data')

    args = parser.parse_args()

    bop_path = args.bop_path
    dataset_name = args.dataset_name
    force_rewrite = args.force_rewrite == 'True'
    is_training_data = args.is_training_data == 'True'
    data_folder = args.data_folder


    generate_GT_images(bop_path, dataset_name, force_rewrite, is_training_data, data_folder)
