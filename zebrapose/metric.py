import numpy as np
import torch
import sys
sys.path.append("../bop_toolkit")
from bop_toolkit_lib import pose_error


def Calculate_ADD_Error_BOP(R_GT,t_GT, R_predict, t_predict, vertices):
    t_GT = t_GT.reshape((3,1))
    t_predict = np.array(t_predict).reshape((3,1))

    return pose_error.add(R_predict, t_predict, R_GT, t_GT, vertices)

def Calculate_ADI_Error_BOP(R_GT,t_GT, R_predict, t_predict, vertices):
    t_GT = t_GT.reshape((3,1))
    t_predict = np.array(t_predict).reshape((3,1))

    return pose_error.adi(R_predict, t_predict, R_GT, t_GT, vertices)
