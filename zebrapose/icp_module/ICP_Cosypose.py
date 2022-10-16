import sys
import os
sys.path.insert(0, os. path. dirname(os.path.abspath(__file__)))

from cosypose_icp.icp_refiner_modified import icp_refinement as icp_refinement_cosypose
import numpy as np
from PIL import Image

from ICP_Render import ICP_Render

def read_depth(path):
    depth = np.asarray(Image.open(path)).copy()
    depth = depth.astype(np.float)
    return depth

class ICPRefiner:
    def __init__(self, obj_mesh, im_width, im_height,  num_iters = 100):
        self.render = ICP_Render(obj_mesh, im_width, im_height)
        self.im_width = im_width
        self.im_height = im_height
        self.num_iters = num_iters
        
    def refine_poses(self, t_est, R_est, mask, depth, cam_K):
        t_est = t_est.flatten()
        object_depth_est = self.render.get_object_depth(t_est, R_est.flatten(), cam_K)

        TCO_pred = np.zeros((4, 4))
        # R_est, t_est is from model to camera
        TCO_pred[3, 3] = 1
        TCO_pred[:3, 3] = t_est / 100.
        TCO_pred[:3, :3] = R_est 
        
        predictions_refined = TCO_pred.copy()

        depth = depth / 1000.
        object_depth_est = object_depth_est / 100.
    
        TCO_refined, retval = icp_refinement_cosypose(
            depth, object_depth_est, mask, cam_K, TCO_pred, n_min_points=1000, n_iters=self.num_iters
        )

        if retval != -1:
            predictions_refined = TCO_refined

        R_refined = predictions_refined[:3, :3]
        t_refined = predictions_refined[:3, 3] * 100.

        return R_refined, t_refined

'''
if __name__ == "__main__":
    
    mesh_path = ''
    depth_path = ''
    im_width = 640
    im_height = 480

    depth_image = read_depth(depth_path)

    R = np.array([0.993969, 0.0777435, -0.0774463, -0.0762715, -0.0179631, -0.996933, -0.0788964, 0.996813, -0.0119239])

    t = np.array([144.364, -132.374, 1163.3])
    t = t*0.1

    t = t + np.array([10, 10, 30])
    intrinsic_matrix = np.zeros((3,3))

    intrinsic_matrix[0,0] = 550.0             #fx
    intrinsic_matrix[1,1] = 540.0            #fy
    intrinsic_matrix[0,2] = 346.0           #cx
    intrinsic_matrix[1,2] = 244.0            #cy
    intrinsic_matrix[2,2] = 1.0 

    mask = np.ones((im_height, im_width))

    icp_refiner = ICPRefiner(mesh_path, im_width, im_height, icp_type, num_iters=10)

    R = np.reshape(R, (3,3))
    import time
    start = time.time()

    R_refined, t_refined = icp_refiner.refine_poses(t, R, mask, depth_image, intrinsic_matrix)

    end = time.time()
    print(end-start)
'''
