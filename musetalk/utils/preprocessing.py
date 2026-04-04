import sys
try:
    from .face_detection import FaceAlignment, LandmarksType
except ImportError:
    from face_detection import FaceAlignment, LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
import torch
from tqdm import tqdm

MMPOSE_AVAILABLE = False
model = None
inference_topdown = None
merge_data_samples = None
try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import merge_data_samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
    if os.path.exists(checkpoint_file):
        model = init_model(config_file, checkpoint_file, device=device)
        MMPOSE_AVAILABLE = True
    else:
        print(f"[MuseTalk] DWPose checkpoint not found at {checkpoint_file}; using face detector fallback (quality may drop).")
except Exception as e:
    print(f"[MuseTalk] mmpose unavailable ({e}); using face detector fallback (quality may drop).")

# initialize the face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device)

# maker if the bbox is not sufficient 
coord_placeholder = (0.0,0.0,0.0,0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_bbox_range(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        # get bounding boxes by face detection
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        if MMPOSE_AVAILABLE:
            results = inference_topdown(model, np.asarray(fb)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark = keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
        else:
            average_range_minus.append(0)
            average_range_plus.append(0)

    if MMPOSE_AVAILABLE and len(average_range_minus) > 0:
        text_range = f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    else:
        text_range = f"Total frame:「{len(frames)}」 mmpose fallback mode (face detector only), current bbox_shift: {upperbondrange}"
    return text_range
    

def get_landmark_and_bbox(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        # get bounding boxes by face detection
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        if MMPOSE_AVAILABLE:
            results = inference_topdown(model, np.asarray(fb)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark = keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)
        else:
            face_land_mark = None

        # adjust the bounding box and append to coordinates list
        for j, f in enumerate(bbox):
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue

            if MMPOSE_AVAILABLE and face_land_mark is not None:
                half_face_coord = face_land_mark[29]
                range_minus = (face_land_mark[30] - face_land_mark[29])[1]
                range_plus = (face_land_mark[29] - face_land_mark[28])[1]
                average_range_minus.append(range_minus)
                average_range_plus.append(range_plus)
                if upperbondrange != 0:
                    # manual adjustment: + shifts down (towards 29), - shifts up (towards 28)
                    half_face_coord[1] = upperbondrange + half_face_coord[1]
                half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
                min_upper_bond = 0
                upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

                f_landmark = (
                    np.min(face_land_mark[:, 0]),
                    int(upper_bond),
                    np.max(face_land_mark[:, 0]),
                    np.max(face_land_mark[:, 1]),
                )
                x1, y1, x2, y2 = f_landmark

                if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                    coords_list += [f]
                    print("error bbox:", f)
                else:
                    coords_list += [f_landmark]
            else:
                x1, y1, x2, y2 = f
                # Fallback path: shift top boundary by bbox_shift while keeping valid box bounds.
                y1 = max(0, int(y1 + upperbondrange))
                if y2 - y1 <= 0 or x2 - x1 <= 0:
                    coords_list += [f]
                else:
                    coords_list += [(int(x1), int(y1), int(x2), int(y2))]
                average_range_minus.append(0)
                average_range_plus.append(0)
    
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    if MMPOSE_AVAILABLE and len(average_range_minus) > 0:
        print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    else:
        print(f"Total frame:「{len(frames)}」 mmpose fallback mode (face detector only), current bbox_shift: {upperbondrange}")
    print("*************************************************************************************************************************************")
    return coords_list,frames
    

if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list,full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)
        
        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)
