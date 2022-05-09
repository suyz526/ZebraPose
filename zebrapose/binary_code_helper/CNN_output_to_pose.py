from binary_code_helper.class_id_encoder_decoder import class_code_images_to_class_id_image
import numpy as np
import cv2
import pyprogressivex

def load_dict_class_id_3D_points(path):
    total_numer_class = 0
    number_of_itration = 0

    dict_class_id_3D_points = {}
    with open(path, "r") as f:
        first_line = f.readline()
        total_numer_class_, divide_number_each_itration, number_of_itration_ = first_line.split(" ") 
        divide_number_each_itration = float(divide_number_each_itration)
        total_numer_class = float(total_numer_class_)
        number_of_itration = float(number_of_itration_)

        for line in f:
            line = line[:-1]
            code, x, y, z= line.split(" ")
            code = float(code)
            x = float(x)
            y = float(y)
            z = float(z)

            dict_class_id_3D_points[code] = np.array([x,y,z])

    return total_numer_class, divide_number_each_itration, number_of_itration, dict_class_id_3D_points

def mapping_pixel_position_to_original_position(pixels, Bbox, Bbox_Size):
    """
    The image was cropped and resized. This function returns the original pixel position
    input:
        pixels: pixel position after cropping and resize, which is a numpy array, N*(X, Y)
        Bbox: Bounding box for the cropping, minx miny width height
    """
    ratio_x = Bbox[2] / Bbox_Size
    ratio_y = Bbox[3] / Bbox_Size

    original_pixel_x = ratio_x*pixels[:,0] + Bbox[0]     
    original_pixel_y = ratio_y*pixels[:,1] + Bbox[1] 

    original_pixel_x = original_pixel_x.astype('int')
    original_pixel_y = original_pixel_y.astype('int')

    return np.concatenate((original_pixel_x.reshape(-1, 1), original_pixel_y.reshape(-1, 1)), 1)


def build_non_unique_2D_3D_correspondence(Pixel_position, class_id_image, dict_class_id_3D_points):
    Point_2D = np.concatenate((Pixel_position[1].reshape(-1, 1), Pixel_position[0].reshape(-1, 1)), 1)   #(npoint x 2)
    
    ids_for_searching = class_id_image[Point_2D[:, 1], Point_2D[:, 0]]

    Points_3D = np.zeros((Point_2D.shape[0],3))
    for i in range(Point_2D.shape[0]):
        if np.isnan(np.array(dict_class_id_3D_points[ids_for_searching[i]])).any():
            continue         
        Points_3D[i] = np.array(dict_class_id_3D_points[ids_for_searching[i]])

    return Point_2D, Points_3D


def build_unique_2D_3D_correspondence(Pixel_position, class_id_image, dict_class_id_3D_points):
    # if multiple 2D pixel match to a 3D vertex. For this vertex, its corres pixel will be the mean position of those pixels

    Point_2D = np.concatenate((Pixel_position[1].reshape(-1, 1), Pixel_position[0].reshape(-1, 1)), 1)   #(npoint x 2)
    ids_for_searching = class_id_image[Point_2D[:, 1], Point_2D[:, 0]]

    #build a dict for 3D points and all 2D pixel
    unique_3D_2D_corres = {}
    for i in range(Point_2D.shape[0]):      
        if ids_for_searching[i] in unique_3D_2D_corres.keys():
            unique_3D_2D_corres[ids_for_searching[i]].append(Point_2D[i])
        else:
            unique_3D_2D_corres[ids_for_searching[i]] = [Point_2D[i]]

    Points_3D = np.zeros((len(unique_3D_2D_corres),3))
    Points_2D = np.zeros((len(unique_3D_2D_corres),2))
    for counter, (key, value) in enumerate(unique_3D_2D_corres.items()):
        Points_3D[counter] = dict_class_id_3D_points[key]
        sum_Pixel_2D = np.zeros((1,2))
        for Pixel_2D in value:
            sum_Pixel_2D = sum_Pixel_2D + Pixel_2D
        unique_Pixel_2D = sum_Pixel_2D / len(value)
        Points_2D[counter] = unique_Pixel_2D

    return Points_2D, Points_3D


def get_class_id_image_validmask(class_id_image):
    mask_image = np.zeros(class_id_image.shape)
    mask_image[class_id_image.nonzero()]=1
    return mask_image


def CNN_outputs_to_object_pose(mask_image, class_code_image, Bbox, Bbox_Size, class_base=2, dict_class_id_3D_points=None, intrinsic_matrix=None):
    if intrinsic_matrix is None:
        intrinsic_matrix = np.zeros((3,3))

        intrinsic_matrix[0,0] = 572.4114             #fx
        intrinsic_matrix[1,1] = 573.57043            #fy
        intrinsic_matrix[0,2] = 325.2611             #cx
        intrinsic_matrix[1,2] = 242.04899            #cy
        intrinsic_matrix[2,2] = 1.0 

    class_id_image = class_code_images_to_class_id_image(class_code_image, class_base)
    Points_2D = mask_image.nonzero()
        
    # find the 2D-3D correspondences and Ransac + PnP
    build_2D_3D_correspondence = build_non_unique_2D_3D_correspondence
    success = False
    rot = []
    tvecs = []

    if Points_2D[0].size != 0:   
        Points_2D,  Points_3D = build_2D_3D_correspondence(Points_2D, class_id_image, dict_class_id_3D_points)
        # PnP needs atleast 6 unique 2D-3D correspondences to run
        # mapping the pixel position to its original position
      
        Original_Points_2D = mapping_pixel_position_to_original_position(Points_2D, Bbox, Bbox_Size)

        if len(Original_Points_2D) >= 6:
            success = True
            coord_2d = np.ascontiguousarray(Original_Points_2D.astype(np.float32))
            coord_3d = np.ascontiguousarray(Points_3D.astype(np.float32))
            intrinsic_matrix = np.ascontiguousarray(intrinsic_matrix)

            if True:
                pose_ests, label = pyprogressivex.find6DPoses(
                                                            x1y1 = coord_2d.astype(np.float64),
                                                            x2y2z2 = coord_3d.astype(np.float64),
                                                            K = intrinsic_matrix.astype(np.float64),
                                                            threshold = 2,  
                                                            neighborhood_ball_radius=20,
                                                            spatial_coherence_weight=0.1,
                                                            maximum_tanimoto_similarity=0.9,
                                                            max_iters=400,
                                                            minimum_point_number=6,
                                                            maximum_model_number=1
                                                        )
                if pose_ests.shape[0] != 0:
                    rot = pose_ests[0:3, :3]
                    tvecs = pose_ests[0:3, 3]
                    tvecs = tvecs.reshape((3,1))
                else:
                    rot = np.zeros((3,3))
                    tvecs = np.zeros((3,1))
                    success = False
            
            else:
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(Points_3D.astype(np.float32),
                                                            Original_Points_2D.astype(np.float32), intrinsic_matrix, distCoeffs=None,
                                                            reprojectionError=2, iterationsCount=150, flags=cv2.SOLVEPNP_EPNP)
                rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
                pred_pose = np.append(rot, tvecs, axis=1)   
    return rot, tvecs, success

