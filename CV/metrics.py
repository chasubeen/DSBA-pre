import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def accuracy(output, target):
    preds = torch.argmax(output, dim = -1)
    counts = (preds == target).sum().item()
    acc = counts / len(target)

    return acc


def top_k_error(output, target, k=5):
    # Top-k 정확도 계산
    top_k_preds = torch.topk(output, k, dim = -1).indices  # Top-k class의 index
    correct = target.view(-1, 1).expand_as(top_k_preds)  # 정답을 Top-k 크기로 확장
    counts = (top_k_preds == correct).any(dim = -1)  # Top-k 내에 정답이 있는지 확인
    top_k_acc = counts.float().mean().item()  

    # Top-k Error = 1 - Top-k Accuracy
    return 1.0 - top_k_acc


## Precision: TP/(TP+FP)
def precision(output, target, average = "weighted"):
    preds = torch.argmax(output, dim = -1).cpu().numpy()     
    target = target.cpu().numpy()
    
    return precision_score(target, preds, average = average, zero_division = 0)


## Recall: TP/(TP+FN)
def recall(output, target, average="weighted"):
    preds = torch.argmax(output, dim = -1).cpu().numpy()
    target = target.cpu().numpy()
    
    return recall_score(target, preds, average = average, zero_division = 0)


## F1-score: 2 * {(precision * recall) / (precision + recall)}
def f1(output, target, average = "weighted"):
    preds = torch.argmax(output, dim = -1).cpu().numpy()
    target = target.cpu().numpy()

    return f1_score(target, preds, average = average, zero_division = 0)