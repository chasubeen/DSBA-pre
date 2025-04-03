from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def apply_scaling(trn, val, tst, method="standard"):
    # scaling 방식 선택
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # scaler 학습
    scaler.fit(trn)

    trn = scaler.transform(trn)
    val = scaler.transform(val)
    tst = scaler.transform(tst)

    return trn, val, tst