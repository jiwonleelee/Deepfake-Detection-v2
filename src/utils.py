import random
import os
import numpy as np
import torch
import csv


def set_seed(seed=42):
    # Python 기반 시드 고정
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch 시드 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN 결정론적 연산 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # [추가] 환경 변수 및 알고리즘 고정
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # 일부 연산 재현성 보장
    # 아래 설정은 일부 모델에서 에러를 낼 수 있으나, 가장 강력한 고정 방법입니다.
    torch.use_deterministic_algorithms(True, warn_only=True)

def log_to_csv(m_name, epoch, train_loss, train_acc, val_loss, video_acc, csv_path):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['model', 'epoch', 'train_loss', 'train_acc', 'val_loss', 'video_acc'])
        writer.writerow([m_name, epoch, train_loss, train_acc, val_loss, video_acc])