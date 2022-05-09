import os
import sys

sys.path.append("../Tools_for_BOP")
import bop_io


def generate_meshs(bop_path, dataset_name, divide_number_each_iteration, number_of_itration, executable_path):
    dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global = bop_io.get_dataset(bop_path,dataset_name)

    if not(os.path.exists(dataset_dir + "/models_GT_color/")):
        os.makedirs(dataset_dir + "/models_GT_color/")
    
    for counter, (m_id,model_ply) in enumerate(model_plys.items()): 
        orginal_mesh = model_ply[:-4]
        orginal_mesh = orginal_mesh + ".obj"
        mesh_fname = model_ply.split("/")[-1]
        txt_fname = "Class_CorresPoint"
        obj_id = mesh_fname[4:-4]
        txt_fname = txt_fname + obj_id + ".txt"

        mesh_fn_write = dataset_dir + "/models_GT_color/" + mesh_fname    
        txt_fname = dataset_dir + "/models_GT_color/" + txt_fname

        executable = "{} {} {} {} {} {}".format(executable_path, divide_number_each_iteration, number_of_itration, orginal_mesh, txt_fname, mesh_fn_write)
        os.system(executable)


if __name__ == "__main__":
    bop_path = sys.argv[1]
    dataset_name = sys.argv[2]
    divide_number_each_iteration = sys.argv[3]
    number_of_itration = sys.argv[4]
    executable_path = sys.argv[5]
    generate_meshs(bop_path, dataset_name, divide_number_each_iteration, number_of_itration, executable_path)
    