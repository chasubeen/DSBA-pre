from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def apply_scaling(trn, val, tst, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'minmax_square':
        scaler = MinMaxScaler()
    elif method == 'minmax_m1p1':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # scaler 학습
    scaler.fit(trn)

    trn_scaled = scaler.transform(trn)
    val_scaled = scaler.transform(val)
    tst_scaled = scaler.transform(tst)

    # 후처리
    if method == 'minmax_square':
        trn_scaled = np.square(trn_scaled)
        val_scaled = np.square(val_scaled)
        tst_scaled = np.square(tst_scaled)

    return trn_scaled, val_scaled, tst_scaled