import torch
from metrics import accuracy, top_k_error, precision, recall, f1
from tqdm import tqdm  # 진행 상태 표시


### 학습 함수
def trainer(model, dataloader, loss, optimizer, device):
    # 학습 모드
    model.train()

    total_loss, total_acc = 0.0, 0.0

    for x, y in tqdm(dataloader, desc = "Training", leave = False):
        x, y = x.to(device), y.to(device)

        ## Forward Pass 
        y_est, _ = model(x)  # Feature Vector 제외하고 출력만 사용
        cost = loss(y_est, y)

        total_loss += cost.item()
        total_acc += accuracy(y_est, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)

    return avg_loss, avg_acc



### 평가 함수
def evaluator(model, dataloader, loss, device):
    # 평가 모드
    model.eval()  

    total_loss, total_acc = 0.0, 0.0
    total_top_k_error, total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc = "Evaluating", leave = False):
            x, y = x.to(device), y.to(device)

            ## Forward Pass
            y_est, _ = model(x)
            cost = loss(y_est, y)

            total_loss += cost.item()

            ## 성능 평가 지표
            total_acc += accuracy(y_est, y)
            total_top_k_error += top_k_error(y_est, y, k = 5)
            total_precision += precision(y_est, y)
            total_recall += recall(y_est, y)
            total_f1 += f1(y_est, y)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    avg_top_k_error = total_top_k_error / len(dataloader)
    avg_precision = total_precision / len(dataloader)
    avg_recall = total_recall / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "top_k_error": avg_top_k_error,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
    }