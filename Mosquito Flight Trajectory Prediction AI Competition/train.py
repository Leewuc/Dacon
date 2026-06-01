"""
DACON 모기 비행 궤적 예측 — 최종 솔루션
CNN + MAE pretraining + Smooth R-Hit + Pseudo-labeling
OOF: 68.02%, LB: 0.6904 (단독) / 0.6906 (블렌드)
"""
import numpy as np, pandas as pd, os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from pathlib import Path

# ── 설정 ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path('data')          # train/, test/, train_labels.csv
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR  = DATA_DIR / 'test'
DT      = 0.04   # 타임스텝 간격 (초)
PRED_DT = 0.08   # 예측 시간 (초)
PSEUDO_W = 0.5   # pseudo-label 가중치
SEEDS    = [42, 7, 2025, 13, 99]
N_FOLD   = 10
CFG = {'ch': 128, 'n': 6, 'dr': 0.3, 'ep1': 80, 'ep2': 40, 'tau_s': 0.003, 'tau_e': 0.0005}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def r_hit(p, t): return float(np.mean(np.linalg.norm(np.asarray(p) - np.asarray(t), axis=-1) <= 0.01))

labels    = pd.read_csv(DATA_DIR / 'train_labels.csv').set_index('id')
train_ids = sorted([f.replace('.csv', '') for f in os.listdir(TRAIN_DIR)])
test_ids  = sorted([f.replace('.csv', '') for f in os.listdir(TEST_DIR)])
X_train   = np.array([pd.read_csv(TRAIN_DIR/f'{t}.csv')[['x','y','z']].values for t in train_ids], dtype=np.float32)
y_train   = np.array([labels.loc[t, ['x','y','z']].values for t in train_ids], dtype=np.float32)
X_test    = np.array([pd.read_csv(TEST_DIR/f'{t}.csv')[['x','y','z']].values  for t in test_ids],  dtype=np.float32)

# pseudo-label: 초기 모델 예측 평균 (정규화 프레임)
# pseudo_y_ten = np.load('pseudo_test_norm.npy')

# ── Azimuth 정규화 ────────────────────────────────────────────────────────────
def azimuth_norm(X, y=None):
    """마지막 속도 방향 → +X 회전, 마지막 위치 → 원점"""
    last = X[:, -1:, :]; X_rel = X - last
    vxy = X[:, -1, :2] - X[:, -2, :2]
    n   = np.linalg.norm(vxy, axis=1, keepdims=True) + 1e-9
    ca, sa = vxy[:, 0] / n[:, 0], vxy[:, 1] / n[:, 0]
    Xn = np.concatenate([
        ca[:,None,None]*X_rel[:,:,0:1] + sa[:,None,None]*X_rel[:,:,1:2],
       -sa[:,None,None]*X_rel[:,:,0:1] + ca[:,None,None]*X_rel[:,:,1:2],
        X_rel[:,:,2:3]], axis=2)
    yn = None
    if y is not None:
        yr = y - last[:,0,:]
        yn = np.stack([ca*yr[:,0]+sa*yr[:,1], -sa*yr[:,0]+ca*yr[:,1], yr[:,2]], axis=1)
    return Xn, yn, ca, sa

def azimuth_denorm(pn, xl, ca, sa):
    return np.stack([ca*pn[:,0]-sa*pn[:,1], sa*pn[:,0]+ca*pn[:,1], pn[:,2]], axis=1) + xl

Xtrn, ytrn, ca_tr, sa_tr = azimuth_norm(X_train, y_train)
Xten, _,    ca_te, sa_te = azimuth_norm(X_test)
cv_trn = (-Xtrn[:,-2,:] * (PRED_DT/DT)).astype(np.float32)  # 등속도 외삽
cv_ten = (-Xten[:,-2,:] * (PRED_DT/DT)).astype(np.float32)

# ── 피처 & 데이터셋 ───────────────────────────────────────────────────────────
def make_feat(X):
    """(N, K, 3) → (N, K, 6): position + velocity"""
    vel = np.diff(X, axis=1) / DT
    vp  = np.concatenate([np.zeros((len(X),1,3), dtype=np.float32), vel], axis=1)
    return np.concatenate([X, vp], axis=2).astype(np.float32)

class DS(Dataset):
    def __init__(self, X, y=None, cv=None, aug=True):
        feat = make_feat(X)
        mf = np.array([1,-1,1,1,-1,1], dtype=np.float32)  # y축 반전
        my = np.array([1,-1,1], dtype=np.float32)
        if aug and y is not None:
            feat = np.concatenate([feat, feat*mf[None,None,:]])
            y    = np.concatenate([y,    y*my])
            cv   = np.concatenate([cv,   cv*my])
        self.X  = torch.from_numpy(feat)
        self.y  = torch.from_numpy(y).float()  if y  is not None else None
        self.cv = torch.from_numpy(cv).float() if cv is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.y[i], self.cv[i]) if self.y is not None else self.X[i]

# ── 모델 ──────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c,c,3,padding=1), nn.BatchNorm1d(c), nn.GELU(),
                                  nn.Conv1d(c,c,3,padding=1), nn.BatchNorm1d(c))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))

class CNN(nn.Module):
    def __init__(self, ch=128, n=6, dr=0.3):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv1d(6,ch,1), nn.BatchNorm1d(ch), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(n)])
        self.head   = nn.Sequential(nn.Linear(ch*2,256), nn.GELU(), nn.Dropout(dr),
                                     nn.Linear(256,64), nn.GELU(), nn.Linear(64,3))
    def forward(self, x):
        x = x.permute(0,2,1); x = self.blocks(self.stem(x))
        return self.head(torch.cat([x.mean(2), x.max(2).values], 1))

# ── 손실 함수 ─────────────────────────────────────────────────────────────────
def smooth_rhit(pred, target, tau):
    """Smooth R-Hit: -sigmoid((0.01 - d) / tau)"""
    return -torch.sigmoid((0.01 - torch.norm(pred - target, dim=-1)) / tau).mean()

# ── 추론 (y-flip TTA) ─────────────────────────────────────────────────────────
@torch.no_grad()
def infer_tta(m, X_norm, cv):
    m.eval()
    def _fwd(X):
        out = []
        for x in DataLoader(DS(X, aug=False), 512, shuffle=False):
            out.append(m((x[0] if isinstance(x, (list,tuple)) else x).to(DEVICE)).cpu().numpy())
        return np.concatenate(out)
    r1 = _fwd(X_norm)
    Xf = X_norm.copy(); Xf[:,:,1] *= -1
    r2 = _fwd(Xf); r2[:,1] *= -1
    return (r1 + r2) / 2 + cv

# ── 학습 루프 ─────────────────────────────────────────────────────────────────
kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=42)
oof_norm_sum  = np.zeros((len(train_ids), 3), dtype=np.float64)
test_norm_sum = np.zeros((len(test_ids),  3), dtype=np.float64)

# pseudo_y_ten = np.load(...)  # 사전 학습 모델의 테스트 예측 (정규화 프레임)

for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    oof_seed = np.zeros((len(train_ids), 3))
    test_seed = np.zeros((len(test_ids), 3))

    for fold, (tri, vali) in enumerate(kf.split(Xtrn)):
        trl_real   = DataLoader(DS(Xtrn[tri], ytrn[tri], cv_trn[tri]), 256, shuffle=True, drop_last=True)
        # trl_pseudo = DataLoader(DS(Xten, pseudo_y_ten, cv_ten), 256, shuffle=True, drop_last=True)

        m   = CNN(**{k: CFG[k] for k in ['ch','n','dr']}).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['ep1'])

        # ── Phase 1: MAE pretraining ──────────────────────────────────────────
        for ep in range(CFG['ep1']):
            m.train()
            for Xb, yb, cvb in trl_real:
                Xb, yb, cvb = Xb.to(DEVICE), yb.to(DEVICE), cvb.to(DEVICE)
                opt.zero_grad()
                loss = F.l1_loss(m(Xb), yb - cvb)
                # pseudo loss: loss += PSEUDO_W * F.l1_loss(m(Xp), yp - cvp)
                loss.backward(); opt.step()
            sch.step()

        p_norm = infer_tta(m, Xtrn[vali], cv_trn[vali])
        best_s = r_hit(azimuth_denorm(p_norm, X_train[vali,-1,:], ca_tr[vali], sa_tr[vali]), y_train[vali])
        best_st = {k: v.clone() for k, v in m.state_dict().items()}

        # ── Phase 2: Smooth R-Hit fine-tuning ────────────────────────────────
        opt2 = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=CFG['ep2'])
        for ep in range(CFG['ep2']):
            tau = CFG['tau_s'] * (CFG['tau_e']/CFG['tau_s'])**(ep/max(CFG['ep2']-1,1))
            m.train()
            for Xb, yb, cvb in trl_real:
                Xb, yb, cvb = Xb.to(DEVICE), yb.to(DEVICE), cvb.to(DEVICE)
                opt2.zero_grad()
                loss = smooth_rhit(m(Xb) + cvb, yb, tau)
                # pseudo loss: loss += PSEUDO_W * smooth_rhit(m(Xp)+cvp, yp, tau)
                loss.backward(); opt2.step()
            sch2.step()
            p_norm = infer_tta(m, Xtrn[vali], cv_trn[vali])
            s = r_hit(azimuth_denorm(p_norm, X_train[vali,-1,:], ca_tr[vali], sa_tr[vali]), y_train[vali])
            if s > best_s: best_s = s; best_st = {k:v.clone() for k,v in m.state_dict().items()}

        m.load_state_dict(best_st)
        oof_seed[vali] = infer_tta(m, Xtrn[vali], cv_trn[vali])
        test_seed     += infer_tta(m, Xten, cv_ten) / N_FOLD
        print(f'Seed{seed} Fold{fold+1}: {r_hit(azimuth_denorm(oof_seed[vali], X_train[vali,-1,:], ca_tr[vali], sa_tr[vali]), y_train[vali]):.5f}', flush=True)

    oof_norm_sum  += oof_seed
    test_norm_sum += test_seed

# ── 앙상블 평균 & 저장 ────────────────────────────────────────────────────────
N = len(SEEDS)
oof_norm  = oof_norm_sum  / N
test_norm = test_norm_sum / N
oof_abs   = azimuth_denorm(oof_norm,  X_train[:,-1,:], ca_tr, sa_tr)
test_abs  = azimuth_denorm(test_norm, X_test[:,-1,:],  ca_te, sa_te)

print(f'Final OOF: {r_hit(oof_abs, y_train):.5f}')

pd.DataFrame({'id': test_ids, 'x': test_abs[:,0], 'y': test_abs[:,1], 'z': test_abs[:,2]}).to_csv(
    'submission.csv', index=False)
