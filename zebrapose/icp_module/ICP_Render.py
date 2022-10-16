import Render
import numpy as np


class ICP_Render:
    def __init__(self, mesh_path, im_width, im_height):    
        self.im_width = im_width 
        self.im_height = im_height
        #intrinsic parameters here only for initialization, will be reseted later
        camera_parameters = np.array([im_width, im_height, 570.0, 570.0, 320, 240])

        # can load multiple 3D mesh into buffer, but load only 1 mesh currently
        self.icp_render = Render
        self.icp_render.init(camera_parameters, 1)
        model_scale = 0.1

        print("bind ", mesh_path, " to the render buffer position", 0)   # if load multiple mesh, replace the 0 as mesh id
        self.icp_render.bind_3D_model(mesh_path, 0, model_scale)

    def __del__(self):
        self.icp_render.delete()

    def get_object_depth(self, t, R, Cam_K):
        camera_parameters_local = np.array([self.im_width, self.im_height, Cam_K[0,0], Cam_K[1,1], Cam_K[0,2], Cam_K[1,2]])
        self.icp_render.reset_camera_params(camera_parameters_local)

        depth_map = self.icp_render.get_object_surface_depth(t, R, 0)    # if load multiple mesh, replace the 0 as mesh id
        return depth_map                  


'''
if __name__ == "__main__":
    # test the icp_render
    
    mesh_path = '/home/ysu/data/data_object_pose/BOP_dataset/lmo/models/obj_000001.ply'
    im_width = 640
    im_height = 480
    render = ICP_Render(mesh_path, im_width, im_height)

    R = np.array([0.79248247, 0.45155233, -0.40996589, 0.47694863, -0.87778542, -0.04486379, -0.38012043, -0.1599789, -0.91099682])

    t = np.array([-71.33676147, -115.092453, 775.05419922])
    t = t*0.1
    intrinsic_matrix = np.zeros((3,3))

    intrinsic_matrix[0,0] = 550.0             #fx
    intrinsic_matrix[1,1] = 540.0            #fy
    intrinsic_matrix[0,2] = 346.0           #cx
    intrinsic_matrix[1,2] = 244.0            #cy
    intrinsic_matrix[2,2] = 1.0 

    import time

    start = time.time()

    test_depth = render.get_object_depth(t, R, intrinsic_matrix)

    end = time.time()
    
    render.icp_render.render_GT_visible_side(t, R, 0, '/home/ysu/test_render_image.png')      

    print(end-start)
'''