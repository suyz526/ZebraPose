import json

        
def get_detection_results(detection_results_file, rgb_fns, obj_id, score_thr):
    with open(detection_results_file) as jsonFile:
        detections = json.load(jsonFile)
        jsonFile.close()
    
    Bbox = [None for x in range(len(rgb_fns))]
    for counter, rgb_fn in enumerate(rgb_fns):
        #rgb_files    ...datasetpath/train/scene_id/rgb/img.png
        rgb_fn = rgb_fn.split("/")
        scene_id = int(rgb_fn[-3])
        img_id = int(rgb_fn[-1][:-4])
        detection_result_key = "{}/{}".format(scene_id,img_id)

        detection = detections[detection_result_key]
        best_det_score = 0
        for d in detection:
            detected_obj_id = d["obj_id"]
            
            bbox_est = d["bbox_est"]  # xywh
            score = d["score"]
            if score < score_thr:
                continue
            if obj_id != detected_obj_id:  # detected obj is not interested
                continue
            if score > best_det_score:
                best_det_score = score
                Bbox[counter] = [int(number) for number in bbox_est]

    return Bbox

def get_detection_scores(detection_results_file, rgb_fns, obj_id, score_thr):
    with open(detection_results_file) as jsonFile:
        detections = json.load(jsonFile)
        jsonFile.close()
    
    scores = [-1 for x in range(len(rgb_fns))]
    for counter, rgb_fn in enumerate(rgb_fns):
        #rgb_files    ...datasetpath/train/scene_id/rgb/img.png
        rgb_fn = rgb_fn.split("/")
        scene_id = int(rgb_fn[-3])
        img_id = int(rgb_fn[-1][:-4])
        detection_result_key = "{}/{}".format(scene_id,img_id)

        detection = detections[detection_result_key]
        best_det_score = 0
        for d in detection:
            detected_obj_id = d["obj_id"]
            
            bbox_est = d["bbox_est"]  # xywh
            score = d["score"]
            if score < score_thr:
                continue
            if obj_id != detected_obj_id:  # detected obj is not interested
                continue
            if score > best_det_score:
                best_det_score = score
                scores[counter] = best_det_score

    return scores


def get_detection_results_vivo(detection_results_file, rgb_fns, obj_id, score_thr):
    with open(detection_results_file) as jsonFile:
        detections = json.load(jsonFile)
        jsonFile.close()
    
    Bbox = {}
    print(len(rgb_fns))
    for counter, rgb_fn in enumerate(rgb_fns):
        #rgb_files    ...datasetpath/train/scene_id/rgb/img.png
        rgb_fn_splited = rgb_fn.split("/")
        scene_id = int(rgb_fn_splited[-3])
        img_id = int(rgb_fn_splited[-1][:-4])
        detection_result_key = "{}/{}".format(scene_id,img_id)

        detection = detections[detection_result_key]
        
        for d in detection:
            detected_obj_id = d["obj_id"]
            
            bbox_est = d["bbox_est"]  # xywh
            score = d["score"]
            if score < score_thr:
                continue
            if obj_id != detected_obj_id:  # detected obj is not interested
                continue
            
            Detected_Bbox = {} 
            Detected_Bbox['bbox_est'] = [int(number) for number in bbox_est]
            Detected_Bbox['score'] = score
            if rgb_fn not in Bbox:  
                Bbox[rgb_fn] = [Detected_Bbox]
            else:
                Bbox[rgb_fn].append(Detected_Bbox)
    return Bbox


def ycbv_select_keyframe(detection_results_file, rgb_fns):
    with open(detection_results_file) as jsonFile:
        detections = json.load(jsonFile)
        jsonFile.close()

    key_frame_idx = []
    for counter, rgb_fn in enumerate(rgb_fns):
        #rgb_files    ...datasetpath/train/scene_id/rgb/img.png
        rgb_fn = rgb_fn.split("/")
        scene_id = int(rgb_fn[-3])
        img_id = int(rgb_fn[-1][:-4])
        detection_result_key = "{}/{}".format(scene_id,img_id)

        if detection_result_key in detections:
            key_frame_idx.append(counter)

    return key_frame_idx

