import torch
import numpy as np

from common_ops import from_output_to_class_mask, from_output_to_class_binary_code
from tools_for_BOP.common_dataset_info import get_obj_info

from binary_code_helper.CNN_output_to_pose import CNN_outputs_to_object_pose

from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP

from tqdm import tqdm

def test_network_with_single_obj(
        net, dataloader, obj_diameter, writer, dict_class_id_3D_points, vertices, step, configs, ignore_n_bit=0,calc_add_and_adi=True):
    
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']
    binary_code_length = configs['binary_code_length']
    divide_number_each_itration = int(configs['divide_number_each_itration'])
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']
    obj_name = configs['obj_name']
    dataset_name=configs['dataset_name']
    
    _, symmetry_obj = get_obj_info(dataset_name)
    
    if obj_name in symmetry_obj:
        Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADD_Error_BOP
        main_metric_name = 'ADI'
        supp_metric_name = 'ADD'
    else:
        Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADI_Error_BOP
        main_metric_name = 'ADD'
        supp_metric_name = 'ADI'
    #net to eval model
    net.eval()

    #test with test data
    ADX_passed=np.zeros(len(dataloader.dataset))
    ADX_error=np.zeros(len(dataloader.dataset))
    AUC_ADX_error=np.zeros(len(dataloader.dataset))
    if calc_add_and_adi:
        ADY_passed=np.zeros(len(dataloader.dataset))
        ADY_error=np.zeros(len(dataloader.dataset))
        AUC_ADY_error=np.zeros(len(dataloader.dataset))

    print("test dataset", flush=True)
    for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(tqdm(dataloader)):
        # do the prediction and get the predicted binary code
        if torch.cuda.is_available():
            data=data.cuda()
            masks = masks.cuda()
            entire_masks = entire_masks.cuda()
            class_code_images = class_code_images.cuda()

        pred_masks_prob, pred_code_prob = net(data)

        pred_codes = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, divided_num_each_interation=divide_number_each_itration, binary_code_length=binary_code_length)
        pred_masks = from_output_to_class_mask(pred_masks_prob)

        # from binary code to pose
        pred_codes = pred_codes.transpose(0, 2, 3, 1)

        pred_masks = pred_masks.transpose(0, 2, 3, 1)
        pred_masks = pred_masks.squeeze(axis=-1)

        Rs = Rs.detach().cpu().numpy()
        ts = ts.detach().cpu().numpy()
        Bboxes = Bboxes.detach().cpu().numpy()
        cam_Ks= cam_Ks.detach().cpu().numpy()
        
        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            if ignore_n_bit!=0 and configs['eval_with_ignore_bits']:
                R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_codes[counter][:,:,:-ignore_n_bit],
                                                                            Bbox, BoundingBox_CropSize_GT, divide_number_each_itration, dict_class_id_3D_points, 
                                                                            intrinsic_matrix=cam_K)
            else:
                R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_codes[counter], 
                                                                            Bbox, BoundingBox_CropSize_GT, divide_number_each_itration, dict_class_id_3D_points, 
                                                                            intrinsic_matrix=cam_K)


            batchsize = dataloader.batch_size
            sample_idx = batch_idx * batchsize + counter
            
            adx_error = 10000
            if success:
                adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
                if np.isnan(adx_error):
                    adx_error = 10000
            if adx_error < obj_diameter*0.1:
                ADX_passed[sample_idx] = 1
            ADX_error[sample_idx] = adx_error
            AUC_ADX_error[counter] = min(100,max(100-adx_error, 0))
            if calc_add_and_adi:
                ady_error = 10000
                if success:
                    ady_error = Calculate_Pose_Error_Supp(r_GT, t_GT, R_predict, t_predict, vertices)
                    if np.isnan(ady_error):
                        ady_error = 10000
                if ady_error < obj_diameter*0.1:
                    ADY_passed[sample_idx] = 1
                ADY_error[sample_idx] = ady_error
                AUC_ADY_error[counter] = min(100,max(100-ady_error, 0))
    
    ADX_passed = np.mean(ADX_passed)
    ADX_error= np.mean(ADX_error)
    AUC_ADX_error = np.mean(AUC_ADX_error)
    writer.add_scalar('TESTDATA_{}/{}_test'.format(main_metric_name,main_metric_name), ADX_passed, step)
    writer.add_scalar('TESTDATA_{}/{}_Error_test'.format(main_metric_name,main_metric_name), ADX_error, step)
    writer.add_scalar('TESTDATA_AUC_{}/AUC_{}_Error_test'.format(main_metric_name,main_metric_name), AUC_ADX_error, step)
    if calc_add_and_adi:
        ADY_passed = np.mean(ADY_passed)
        ADY_error= np.mean(ADY_error)
        AUC_ADY_error = np.mean(AUC_ADY_error)
        writer.add_scalar('TESTDATA_{}/{}_test'.format(supp_metric_name,supp_metric_name), ADY_passed, step)
        writer.add_scalar('TESTDATA_{}/{}_Error_test'.format(supp_metric_name,supp_metric_name), ADY_error, step)
        writer.add_scalar('TESTDATA_AUC_{}/AUC_{}_Error_test'.format(supp_metric_name,supp_metric_name), AUC_ADY_error, step)

    #net back to train mode
    net.train()

    return ADX_passed