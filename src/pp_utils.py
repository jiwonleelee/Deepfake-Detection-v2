import cv2
import numpy as np
import os
import json
import urllib.request
from tqdm import tqdm

def get_aligned_face(image, face_info, target_size=(512, 512)):
    try:
        h_img, w_img = image.shape[:2]
        landmarks = getattr(face_info, 'kps', None)
        bbox = getattr(face_info, 'bbox', None)
        if bbox is None: return None

        x1, y1, x2, y2 = bbox.astype(int)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        if landmarks is not None and len(landmarks) >= 2:
            left_eye, right_eye = landmarks[0], landmarks[1]
            dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            M = cv2.getRotationMatrix2D((float(center_x), float(center_y)), angle, 1.0)
            image = cv2.warpAffine(image, M, (w_img, h_img), flags=cv2.INTER_CUBIC)

        w_box, h_box = x2 - x1, y2 - y1
        side_length = int(max(w_box, h_box) * 2.0)
        half_side = side_length // 2
        nx1, ny1 = center_x - half_side, center_y - half_side
        nx2, ny2 = center_x + half_side, center_y + half_side

        pad_x1, pad_y1 = max(0, -nx1), max(0, -ny1)
        pad_x2, pad_y2 = max(0, nx2 - w_img), max(0, ny2 - h_img)
        nx1, ny1 = max(0, nx1), max(0, ny1)
        nx2, ny2 = min(w_img, nx2), min(h_img, ny2)

        face_crop = image[ny1:ny2, nx1:nx2]
        if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
            face_crop = cv2.copyMakeBorder(face_crop, pad_y1, pad_y2, pad_x1, pad_x2,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return cv2.resize(face_crop, target_size)
    except:
        return None

def get_id_matched_dataset(FILELIST_URL, MANIPULATIONS):
    real_list = [f"{i:03d}.mp4" for i in range(1000)]
    with urllib.request.urlopen(FILELIST_URL) as url:
        file_pairs = json.loads(url.read().decode())
    fake_dict = {m: [] for m in MANIPULATIONS}
    for pair in file_pairs:
        id_a, id_b = pair[0], pair[1]
        for idx_id in [id_a, id_b]:
            try:
                val = int(idx_id)
                if val < 1000:
                    method = MANIPULATIONS[min(val // 250, 3)]
                    name = f"{id_a}_{id_b}.mp4" if idx_id == id_a else f"{id_b}_{id_a}.mp4"
                    if name not in fake_dict[method]: fake_dict[method].append(name)
            except: pass
    for m in MANIPULATIONS: fake_dict[m].sort()
    return real_list, fake_dict