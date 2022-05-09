import numpy as np
import os

#scene_id,im_id,obj_id,score,R,t,time

def write_cvs(evaluation_result_path, filename, obj_id, scene_id_, img_id_, r_, t_, scores):
    filename = os.path.join(evaluation_result_path, filename + '.csv')
    f = open(filename, "w")
    f.write("scene_id,im_id,obj_id,score,R,t,time\n")

    for scene_id, img_id, r, t, score in zip(scene_id_, img_id_, r_, t_, scores):
        if score == -1:
            continue
        r11 = r[0][0]
        r12 = r[0][1]
        r13 = r[0][2]

        r21 = r[1][0]
        r22 = r[1][1]
        r23 = r[1][2]

        r31 = r[2][0]
        r32 = r[2][1]
        r33 = r[2][2]

        f.write(str(scene_id))
        f.write(",")
        f.write(str(img_id))
        f.write(",")
        f.write(str(obj_id))
        f.write(",")
        f.write(str(score)) # score
        f.write(",")
        # R
        f.write(str(r11))
        f.write(" ")
        f.write(str(r12))
        f.write(" ")
        f.write(str(r13))
        f.write(" ")
        f.write(str(r21))
        f.write(" ")
        f.write(str(r22))
        f.write(" ")
        f.write(str(r23))
        f.write(" ")
        f.write(str(r31))
        f.write(" ")
        f.write(str(r32))
        f.write(" ")
        f.write(str(r33))
        f.write(",")
        #t
        f.write(str(t[0][0]))
        f.write(" ")
        f.write(str(t[1][0]))
        f.write(" ")
        f.write(str(t[2][0]))
        f.write(",")
        #time
        f.write("-1\n")
    f.close()