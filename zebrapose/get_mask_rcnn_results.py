import json


def get_detection_results(detection_results_file, rgb_fns, obj_id, score_thr):
    with open(detection_results_file) as jsonFile:
        detection_results = json.load(jsonFile)
        jsonFile.close()
    detections = dict()
    for item in detection_results:
        detection_result_key = "{}/{}".format(item['scene_id'], item['image_id'])
        detection_result_value = dict()
        detection_result_value['obj_id'] = item['category_id']
        detection_result_value['bbox_est'] = item['bbox']
        detection_result_value['score'] = item['score']
        detection_result_value['time'] = item['time']
        detection_result_value['segmentation'] = dict()
        detection_result_value['segmentation']['counts'] = item['segmentation']['counts']
        detection_result_value['segmentation']['size'] = item['segmentation']['size']
        if detection_result_key not in detections.keys():
            detections[detection_result_key]=list()
        detections[detection_result_key].append(detection_result_value)

    Bbox = [None for x in range(len(rgb_fns))]
    Segmentation = [None for x in range(len(rgb_fns))]
    for counter, rgb_fn in enumerate(rgb_fns):
        # rgb_files    ...datasetpath/train/scene_id/rgb/img.png
        rgb_fn = rgb_fn.split("/")
        scene_id = int(rgb_fn[-3])
        img_id = int(rgb_fn[-1][:-4])
        detection_result_key = "{}/{}".format(scene_id, img_id)

        if detection_result_key in detections:
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
                    Segmentation[counter] = d['segmentation']

    return Bbox, Segmentation


def get_detection_scores(detection_results_file, rgb_fns, obj_id, score_thr):
    with open(detection_results_file) as jsonFile:
        detection_results = json.load(jsonFile)
        jsonFile.close()
    detections = dict()
    for item in detection_results:
        detection_result_key = "{}/{}".format(item['scene_id'], item['image_id'])
        detection_result_value = dict()
        detection_result_value['obj_id'] = item['category_id']
        detection_result_value['bbox_est'] = item['bbox']
        detection_result_value['score'] = item['score']
        detection_result_value['time'] = item['time']
        detection_result_value['segmentation'] = dict()
        detection_result_value['segmentation']['counts'] = item['segmentation']['counts']
        detection_result_value['segmentation']['size'] = item['segmentation']['size']
        if detection_result_key not in detections.keys():
            detections[detection_result_key]=list()
        detections[detection_result_key].append(detection_result_value)

    scores = [-1 for x in range(len(rgb_fns))]
    for counter, rgb_fn in enumerate(rgb_fns):
        # rgb_files    ...datasetpath/train/scene_id/rgb/img.png
        rgb_fn = rgb_fn.split("/")
        scene_id = int(rgb_fn[-3])
        img_id = int(rgb_fn[-1][:-4])
        detection_result_key = "{}/{}".format(scene_id, img_id)

        if detection_result_key in detections:
            detection = detections[detection_result_key]
            best_det_score = 0
            for d in detection:
                detected_obj_id = d["obj_id"]
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
    # TODO this function hasn't implemented to output segmention
    # transform maskRCNN json file to fcos json format
    with open(detection_results_file) as jsonFile:
        detection_results = json.load(jsonFile)
        jsonFile.close()
    detections = dict()
    for item in detection_results:
        detection_result_key = "{}/{}".format(item['scene_id'], item['image_id'])
        detection_result_value = dict()
        detection_result_value['obj_id'] = item['category_id']
        detection_result_value['bbox_est'] = item['bbox']
        detection_result_value['score'] = item['score']
        detection_result_value['time'] = item['time']
        detection_result_value['segmentation'] = dict()
        detection_result_value['segmentation']['counts'] = item['segmentation']['counts']
        detection_result_value['segmentation']['size'] = item['segmentation']['size']
        if detection_result_key not in detections.keys():
            detections[detection_result_key]=list()
        detections[detection_result_key].append(detection_result_value)

    Bbox = {}
    print(len(rgb_fns))
    for counter, rgb_fn in enumerate(rgb_fns):
        # rgb_files    ...datasetpath/train/scene_id/rgb/img.png
        rgb_fn_splited = rgb_fn.split("/")
        scene_id = int(rgb_fn_splited[-3])
        img_id = int(rgb_fn_splited[-1][:-4])
        detection_result_key = "{}/{}".format(scene_id, img_id)

        if detection_result_key in detections:
            detection = detections[detection_result_key]
        else:
            continue

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
            Detected_Bbox['segmentation'] = d['segmentation']
            if rgb_fn not in Bbox:
                Bbox[rgb_fn] = [Detected_Bbox]
            else:
                Bbox[rgb_fn].append(Detected_Bbox)
    return Bbox


def ycbv_select_keyframe(detection_results_file, rgb_fns):
    with open(detection_results_file) as jsonFile:
        detection_results = json.load(jsonFile)
        jsonFile.close()
    detections = dict()
    for item in detection_results:
        detection_result_key = "{}/{}".format(item['scene_id'], item['image_id'])
        detection_result_value = dict()
        detection_result_value['obj_id'] = item['category_id']
        detection_result_value['bbox_est'] = item['bbox']
        detection_result_value['score'] = item['score']
        detection_result_value['time'] = item['time']
        detection_result_value['segmentation'] = dict()
        detection_result_value['segmentation']['counts'] = item['segmentation']['counts']
        detection_result_value['segmentation']['size'] = item['segmentation']['size']
        if detection_result_key not in detections.keys():
            detections[detection_result_key]=list()
        detections[detection_result_key].append(detection_result_value)

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

