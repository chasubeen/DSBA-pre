import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def accuracy(output, target):
    with torch.no_grad():
        preds = torch.argmax(output, dim = -1)
        correct = (preds == target).sum().item()
        acc = correct / target.size(0)
    return acc


def top_k_error(output, target, k=5):
    with torch.no_grad():
        batch_size = target.size(0)
        _, top_pred = output.topk(k, dim = -1) # Top-k class의 index
        correct = top_pred.eq(target.view(-1, 1).expand_as(top_pred))  # 정답을 Top-k 크기로 확장
        correct_k = correct.any(dim = 1).float().sum().item()  # Top-k 내에 정답이 있는지 확인
        acc_k = correct_k / batch_size
    # Top-k Error = 1 - Top-k Accuracy    
    return 1.0 - acc_k  


def precision(output, target, average = "weighted"):
    with torch.no_grad():
        preds = torch.argmax(output, dim = -1).cpu()
        target = target.cpu()
        return precision_score(target, preds, average = average, zero_division = 0)


def recall(output, target, average = "weighted"):
    with torch.no_grad():
        preds = torch.argmax(output, dim = -1).cpu()
        target = target.cpu()
        return recall_score(target, preds, average = average, zero_division = 0)


def f1(output, target, average = "weighted"):
    with torch.no_grad():
        preds = torch.argmax(output, dim = -1).cpu()
        target = target.cpu()
        return f1_score(target, preds, average = average, zero_division = 0)