import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels, _ in tqdm(dataloader, desc="  Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), correct / total

def validate_video_level(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    video_scores, video_labels = defaultdict(list), {}
    with torch.no_grad():
        for inputs, labels, video_ids in tqdm(dataloader, desc="  Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item()
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels_np = labels.cpu().numpy()
            for i in range(len(video_ids)):
                vid = video_ids[i]
                video_scores[vid].append(probs[i])
                video_labels[vid] = labels_np[i]
    correct = 0
    for vid in video_scores:
        if (np.mean(video_scores[vid]) > 0.5) == video_labels[vid]: correct += 1
    return running_loss / len(dataloader), correct / len(video_scores)

def save_checkpoint(state, is_best, model_name, save_dir):
    """
    state: epoch, model_state, optimizer_state, best_acc, es_state(추가됨)를 포함한 딕셔너리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 일반 체크포인트 저장 (매 에포크 갱신)
    filename = os.path.join(save_dir, f"{model_name}_checkpoint.pth.tar")
    torch.save(state, filename)
    
    # 2. 베스트 모델 저장 (성능 갱신 시에만)
    if is_best:
        best_filename = os.path.join(save_dir, f"{model_name}_best.pth.tar")
        torch.save(state, best_filename)
        print(f"⭐ [Best Updated] {model_name} - Best Acc: {state['best_acc']:.4f} 저장 완료")
