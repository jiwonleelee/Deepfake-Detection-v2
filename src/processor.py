import os
import cv2
import tempfile
import shutil
import numpy as np
from tqdm import tqdm
import zipfile

def process_video_list(video_list, save_base_path, label_type, detector, pp_utils, 
                       num_frames=15, target_size=(512, 512), margin_rate=1.0, 
                       zip_obj=None, server_url=None, sub_path=None):
                       
    #다양한 출처(Zip, Server)의 영상 리스트를 전처리하는 범용 함수

    for file_path in tqdm(video_list, desc=f"Processing {label_type}"):
        # 파일명 및 저장 경로 설정
        video_name = os.path.splitext(os.path.basename(file_path))[0]
        save_dir = os.path.join(save_base_path, label_type, video_name)
        
        # 이어하기 로직
        if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= num_frames:
            continue
        os.makedirs(save_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            try:
                # 데이터 출처에 따른 로드 로직
                if zip_obj: # Zip 파일인 경우 (CDF 등)
                    tmp.write(zip_obj.read(file_path))
                    tmp.flush()
                    video_source = tmp.name
                elif server_url: # 서버 URL인 경우 (FF++ 등)
                    import urllib.request
                    video_url = server_url + sub_path + file_path
                    urllib.request.urlretrieve(video_url, tmp.name)
                    video_source = tmp.name
                else: # 로컬 경로인 경우
                    video_source = file_path

                cap = cv2.VideoCapture(video_source)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    cap.release()
                    continue

                indices = np.linspace(0, total - 1, num_frames, dtype=int)
                face_crops_temp = []
                frames_cache = []
                last_face_info = None
                video_faces_found = False

                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        face_crops_temp.append(None)
                        continue

                    frames_cache.append(frame)
                    current_crop = None
                    faces = detector.get(frame)

                    if faces:
                        # 면적이 가장 큰 얼굴 선택
                        used_info = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
                        current_crop = pp_utils.get_aligned_face(frame, used_info, target_size=target_size, margin_rate=margin_rate)
                        
                        if current_crop is not None:
                            if not video_faces_found and len(face_crops_temp) > 0:
                                for i in range(len(face_crops_temp)):
                                    retro = pp_utils.get_aligned_face(frames_cache[i], used_info, target_size=target_size, margin_rate=margin_rate)
                                    face_crops_temp[i] = retro if retro is not None else current_crop
                            last_face_info = used_info
                            video_faces_found = True
                    
                    if current_crop is None and last_face_info is not None:
                        current_crop = pp_utils.get_aligned_face(frame, last_face_info, target_size=target_size, margin_rate=margin_rate)
                    
                    face_crops_temp.append(current_crop)

                cap.release()

                # 저장 및 실패 시 삭제
                if video_faces_found and all(c is not None for c in face_crops_temp):
                    for i, crop in enumerate(face_crops_temp):
                        cv2.imwrite(os.path.join(save_dir, f"f{i:03d}_{label_type.lower()}.jpg"), crop)
                else:
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)

            except Exception as e:
                print(f"❌ Error {video_name}: {e}")