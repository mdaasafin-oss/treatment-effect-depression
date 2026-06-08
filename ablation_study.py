#Ablation study
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED CONFIG BASE
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = 'data_generated.csv'
SEED      = 42

# Architecture
ARCH = dict(
    input_dim      = 18,
    encoder_hidden = [256, 128],
    ds             = 64,
    dc             = 64,
    n_treatments   = 6,
    treatment_emb  = 16,
    prop_hidden    = 64,
    outcome_hidden = [128, 64],
    ite_hidden     = 64,
    T_diff         = 100,
    beta_start     = 1e-4,
    beta_end       = 0.02,
    step_emb_dim   = 32,
    diff_hidden    = [128, 128, 128],
)

# Training — shared defaults 
TRAIN_BASE = dict(
    batch_size       = 32,
    max_epochs_s1    = 100,
    max_epochs_s2    = 100,
    patience         = 15,
    weight_decay     = 1e-5,
    grad_clip        = 3.0,   
    lambda_stab      = 1.0,
    lambda_prop      = 1.0,
    lambda_bal       = 0.5,
    lambda_ite       = 0.1,
    lambda_reg       = 0.01,
    vicreg_v         = 25.0,
    vicreg_c         = 1.0,
    prop_clip_eps    = 0.10,
    winsor_pct       = 1.0,
    mmd_sigma        = 1.0,
    noise_sigma      = 0.01,
    feat_drop_p      = 0.10,
)

# Five experiment configurations 
# Only the rows that DIFFER from baseline are listed
EXPERIMENTS = {
    'Exp0_Baseline': dict(
        lr=1e-3, dropout=0.1, diff_dropout=0.0,
        s2_lr=1e-3, use_adamw_cosine=False,
        use_tta=False, tta_n_views=1,
    ),
    'ExpA_HigherLR': dict(
        lr=5e-3, dropout=0.1, diff_dropout=0.0,       # only LR changes
        s2_lr=1e-3, use_adamw_cosine=False,
        use_tta=False, tta_n_views=1,
    ),
    'ExpB_NoDropout': dict(
        lr=1e-3, dropout=0.0, diff_dropout=0.0,        # only dropout changes
        s2_lr=1e-3, use_adamw_cosine=False,
        use_tta=False, tta_n_views=1,
    ),
    'ExpC_AdamWCos': dict(
        lr=1e-3, dropout=0.1, diff_dropout=0.0,        # only S2 optimizer changes
        s2_lr=1e-3, use_adamw_cosine=True,
        use_tta=False, tta_n_views=1,
    ),
    'ExpD_Optimized': dict(
        lr=5e-3, dropout=0.0, diff_dropout=0.0,        # all changes + TTA
        s2_lr=1e-3, use_adamw_cosine=True,
        use_tta=True, tta_n_views=5,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  UTILS
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def make_alphas_bar(T_diff, beta_start, beta_end, device):
    betas      = torch.linspace(beta_start, beta_end, T_diff, device=device)
    alphas_bar = torch.cumprod(1.0 - betas, dim=0)
    return alphas_bar


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────
HAMD_COLS = [f'HAMD{str(i).zfill(2)}' for i in range(1, 18)]
FEAT_COLS = ['VISIT'] + HAMD_COLS
DROP_COLS = [
    'RAW_ID','THERAPY_STATUS','PROTOCOL','THERPHAS','THERCODE',
    'THERAPY1','THERAPY2','THERCOD1','THERCOD2','AGE','GENDER','ORIGIN','GEOCODE',
]


def load_data(path, seed=42):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df[HAMD_COLS] = df[HAMD_COLS] - 1

    agg = {c: 'mean' for c in HAMD_COLS}
    agg['THERAPY'] = 'first'
    df = df.groupby(['UNIQUEID', 'VISIT'], as_index=False).agg(agg)
    df = df.sort_values(['UNIQUEID', 'VISIT']).reset_index(drop=True)

    df['HAMD_TOTAL'] = df[HAMD_COLS].sum(axis=1)
    df['y'] = df.groupby('UNIQUEID')['HAMD_TOTAL'].shift(-1)
    df = df.dropna(subset=['y']).reset_index(drop=True)

    patients = df['UNIQUEID'].unique()
    train_pts, temp = train_test_split(patients, test_size=0.30, random_state=seed)
    val_pts, test_pts = train_test_split(temp, test_size=0.50, random_state=seed)

    train_df = df[df['UNIQUEID'].isin(train_pts)].copy()
    val_df   = df[df['UNIQUEID'].isin(val_pts)].copy()
    test_df  = df[df['UNIQUEID'].isin(test_pts)].copy()

    scaler = StandardScaler()
    train_df[FEAT_COLS] = scaler.fit_transform(train_df[FEAT_COLS])
    val_df[FEAT_COLS]   = scaler.transform(val_df[FEAT_COLS])
    test_df[FEAT_COLS]  = scaler.transform(test_df[FEAT_COLS])
    for s in [train_df, val_df, test_df]:
        s[FEAT_COLS] = s[FEAT_COLS].fillna(0)

    therapy_list = sorted(train_df['THERAPY'].unique())
    therapy_map  = {t: i for i, t in enumerate(therapy_list)}
    unk_idx      = len(therapy_map)
    for s in [train_df, val_df, test_df]:
        s['T'] = s['THERAPY'].map(therapy_map).fillna(unk_idx).astype(int)

    baseline_t = int(train_df['T'].value_counts().idxmax())
    return train_df, val_df, test_df, therapy_map, baseline_t


class DepressionDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(df[FEAT_COLS].values, dtype=torch.float32)
        self.T = torch.tensor(df['T'].values,        dtype=torch.long)
        self.y = torch.tensor(df['y'].values,        dtype=torch.float32)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.T[i], self.y[i]


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
def _mlp_block(in_d, out_d, dropout):
    return nn.Sequential(
        nn.Linear(in_d, out_d), nn.ReLU(), nn.BatchNorm1d(out_d), nn.Dropout(dropout)
    )


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, ds, dc, dropout):
        super().__init__()
        assert hidden_dims[-1] == ds + dc
        self.ds = ds
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers.append(_mlp_block(in_d, h, dropout))
            in_d = h
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        h = self.backbone(x)
        return h[:, :self.ds], h[:, self.ds:]


class PropensityHead(nn.Module):
    def __init__(self, dc, hidden_dim, n_treatments):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dc, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_treatments)
        )
    def forward(self, C): return self.net(C)


class OutcomeModel(nn.Module):
    def __init__(self, ds, dc, n_treatments, t_emb_dim, hidden_dims, dropout):
        super().__init__()
        self.t_embed = nn.Embedding(n_treatments + 1, t_emb_dim)
        in_d, layers = ds + dc + t_emb_dim, []
        for h in hidden_dims:
            layers.append(_mlp_block(in_d, h, dropout))
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, S, C, T):
        return self.net(torch.cat([S, C, self.t_embed(T)], dim=-1)).squeeze(-1)


class ITEHead(nn.Module):
    def __init__(self, ds, dc, hidden_dim, n_treatments):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ds + dc, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_treatments)
        )
    def forward(self, S, C): return self.net(torch.cat([S, C], dim=-1))


class DiffusionDenoiser(nn.Module):
    def __init__(self, dc, ds, n_treatments, t_emb_dim, step_emb_dim, hidden_dims, dropout=0.0):
        super().__init__()
        self.step_embed = nn.Sequential(
            nn.Linear(1, step_emb_dim), nn.SiLU(), nn.Linear(step_emb_dim, step_emb_dim)
        )
        self.t_embed = nn.Embedding(n_treatments + 1, t_emb_dim)
        in_d, layers = dc + ds + step_emb_dim + t_emb_dim, []
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = h
        layers.append(nn.Linear(in_d, dc))
        self.net = nn.Sequential(*layers)

    def forward(self, C_noisy, step, S, T):
        step_emb = self.step_embed(step.float().unsqueeze(-1) / 100.0)
        return self.net(torch.cat([C_noisy, S, step_emb, self.t_embed(T)], dim=-1))


def build_model(exp_cfg, device):
    a = ARCH
    dropout      = exp_cfg['dropout']
    diff_dropout = exp_cfg['diff_dropout']

    encoder    = Encoder(a['input_dim'], a['encoder_hidden'], a['ds'], a['dc'], dropout)
    propensity = PropensityHead(a['dc'], a['prop_hidden'], a['n_treatments'])
    outcome    = OutcomeModel(a['ds'], a['dc'], a['n_treatments'],
                               a['treatment_emb'], a['outcome_hidden'], dropout)
    ite        = ITEHead(a['ds'], a['dc'], a['ite_hidden'], a['n_treatments'])
    denoiser   = DiffusionDenoiser(a['dc'], a['ds'], a['n_treatments'],
                                    a['treatment_emb'], a['step_emb_dim'],
                                    a['diff_hidden'], diff_dropout)

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder    = encoder
            self.propensity = propensity
            self.outcome    = outcome
            self.ite        = ite
            self.denoiser   = denoiser

    return FullModel().to(device)


# ─────────────────────────────────────────────────────────────────────────────
#  LOSSES
# ─────────────────────────────────────────────────────────────────────────────
def _rbf_kernel(X, Y, sigma=1.0):
    XX   = (X**2).sum(1, keepdim=True)
    YY   = (Y**2).sum(1, keepdim=True)
    dist = XX + YY.t() - 2.0*(X @ Y.t())
    return torch.exp(-dist / (2.0*sigma**2))


def mmd_loss(C, T, n_treatments, sigma=1.0):
    groups = [C[T==t] for t in range(n_treatments)]
    total, count = torch.tensor(0.0, device=C.device), 0
    for i in range(n_treatments):
        for j in range(i+1, n_treatments):
            Ci, Cj = groups[i], groups[j]
            if len(Ci)<2 or len(Cj)<2: continue
            kxx = _rbf_kernel(Ci, Ci, sigma).mean()
            kyy = _rbf_kernel(Cj, Cj, sigma).mean()
            kxy = _rbf_kernel(Ci, Cj, sigma).mean()
            total += kxx + kyy - 2.0*kxy
            count += 1
    return total / max(count, 1)


def vicreg_reg(S, v_weight=25.0, c_weight=1.0, gamma=1.0):
    B, d       = S.shape
    S_c        = S - S.mean(dim=0)
    v_loss     = torch.mean(F.relu(gamma - S_c.std(dim=0)))
    cov        = (S_c.T @ S_c) / (B-1)
    mask       = ~torch.eye(d, dtype=torch.bool, device=S.device)
    c_loss     = (cov**2)[mask].mean()
    return v_weight*v_loss + c_weight*c_loss


def _dr_pseudo_single(S, C, T, y, outcome_model, prop_logits, n_treatments, clip_eps, device):
    with torch.no_grad():
        probs = torch.softmax(prop_logits, dim=-1).clamp(clip_eps, 1-clip_eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        y_hat = outcome_model(S, C, T)
        residual = y - y_hat
        Y_dr = torch.zeros(S.shape[0], n_treatments, device=device)
        for t in range(n_treatments):
            T_cf       = torch.full((S.shape[0],), t, dtype=torch.long, device=device)
            mu_t       = outcome_model(S, C, T_cf)
            ipw        = (T == t).float() / probs[:, t]
            Y_dr[:, t] = mu_t + ipw*residual
    return Y_dr


def compute_dr_crossfit(S, C, T, y, outcome_model, propensity_head,
                         baseline_t, n_treatments, clip_eps, winsor_pct, device):
    B, mid = S.shape[0], S.shape[0]//2
    tau    = torch.zeros(B, n_treatments, device=device)
    outcome_model.eval()
    with torch.no_grad():
        tau[:mid] = _dr_pseudo_single(S[:mid], C[:mid], T[:mid], y[:mid],
                                       outcome_model, propensity_head(C[mid:]),
                                       n_treatments, clip_eps, device)
        tau[mid:] = _dr_pseudo_single(S[mid:], C[mid:], T[mid:], y[mid:],
                                       outcome_model, propensity_head(C[:mid]),
                                       n_treatments, clip_eps, device)
    outcome_model.train()
    tau = tau - tau[:, baseline_t:baseline_t+1]
    if winsor_pct < 50:
        lo = torch.tensor(np.percentile(tau.cpu().numpy(), winsor_pct,       axis=0), device=device)
        hi = torch.tensor(np.percentile(tau.cpu().numpy(), 100-winsor_pct,   axis=0), device=device)
        tau = tau.clamp(min=lo, max=hi)
    return tau


def diffusion_loss_fn(denoiser, C0, S, T, alphas_bar, T_diff, device):
    B   = C0.shape[0]
    tau = torch.randint(1, T_diff+1, (B,), device=device)
    ab  = alphas_bar[tau-1].unsqueeze(-1)
    eps = torch.randn_like(C0)
    C_tau = torch.sqrt(ab)*C0 + torch.sqrt(1-ab)*eps
    return F.mse_loss(denoiser(C_tau, tau, S, T), eps)


def stage1_loss_fn(batch, model, baseline_t, cfg_merged, device):
    X, T, y = [v.to(device) for v in batch]
    ns, fd  = cfg_merged['noise_sigma'], cfg_merged['feat_drop_p']

    def augment(x):
        return (x + torch.randn_like(x)*ns) * (torch.rand_like(x) > fd).float()

    S1, C1 = model.encoder(augment(X))
    S2, _  = model.encoder(augment(X))

    y_pred      = model.outcome(S1, C1, T)
    prop_logits = model.propensity(C1)
    ite_pred    = model.ite(S1, C1)
    tau_target  = compute_dr_crossfit(
        S1.detach(), C1.detach(), T, y,
        model.outcome, model.propensity, baseline_t,
        cfg_merged['n_treatments'], cfg_merged['prop_clip_eps'],
        cfg_merged['winsor_pct'], device)

    L_out  = F.mse_loss(y_pred, y)
    L_stab = F.mse_loss(S1, S2)
    L_prop = F.cross_entropy(prop_logits, T)
    L_bal  = mmd_loss(C1, T, cfg_merged['n_treatments'], cfg_merged['mmd_sigma'])
    L_ite  = F.mse_loss(ite_pred, tau_target)
    L_reg  = vicreg_reg(S1, cfg_merged['vicreg_v'], cfg_merged['vicreg_c'])

    total = (L_out
             + cfg_merged['lambda_stab'] * L_stab
             + cfg_merged['lambda_prop'] * L_prop
             + cfg_merged['lambda_bal']  * L_bal
             + cfg_merged['lambda_ite']  * L_ite
             + cfg_merged['lambda_reg']  * L_reg)

    return total, {
        'out': L_out.item(), 'stab': L_stab.item(), 'prop': L_prop.item(),
        'bal': L_bal.item(), 'ite':  L_ite.item(),  'reg':  L_reg.item(),
        'total': total.item()
    }


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_stage1(model, train_ds, val_ds, baseline_t, cfg_merged, device):
    train_loader = DataLoader(train_ds, batch_size=cfg_merged['batch_size'],
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg_merged['batch_size'], shuffle=False)  # matches ultimate model

    params = (list(model.encoder.parameters())
            + list(model.propensity.parameters())
            + list(model.outcome.parameters())
            + list(model.ite.parameters()))

    opt = torch.optim.Adam(params, lr=cfg_merged['lr'],
                           weight_decay=cfg_merged['weight_decay'])
    best_val, patience_ctr, best_state = float('inf'), 0, None

    for epoch in range(1, cfg_merged['max_epochs_s1']+1):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            loss, _ = stage1_loss_fn(batch, model, baseline_t, cfg_merged, device)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg_merged['grad_clip'])
            opt.step()

        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for X, T, y in val_loader:
                X, T, y = X.to(device), T.to(device), y.to(device)
                S, C    = model.encoder(X)
                val_mse += ((model.outcome(S, C, T)-y)**2).mean().item()
        val_mse /= len(val_loader)

        if val_mse < best_val - 1e-5:
            best_val     = val_mse
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg_merged['patience']:
                print(f"    S1 early stop epoch {epoch}, best val MSE={best_val:.4f}")
                break

    model.load_state_dict(best_state)
    print(f"    Stage 1 best val MSE: {best_val:.4f}")
    return best_val


def train_stage2(model, train_ds, val_ds, cfg_merged, exp_cfg, device):
    train_loader = DataLoader(train_ds, batch_size=cfg_merged['batch_size'],
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg_merged['batch_size'], shuffle=False)  # matches ultimate model

    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()

    alphas_bar = make_alphas_bar(ARCH['T_diff'], ARCH['beta_start'], ARCH['beta_end'], device)

    if exp_cfg['use_adamw_cosine']:
        opt = torch.optim.AdamW(model.denoiser.parameters(),
                                lr=exp_cfg['s2_lr'],
                                weight_decay=cfg_merged['weight_decay'])
        total_steps  = len(train_loader) * cfg_merged['max_epochs_s2']
        warmup_steps = len(train_loader) * 3
        scheduler    = cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    else:
        opt       = torch.optim.Adam(model.denoiser.parameters(),
                                     lr=exp_cfg['s2_lr'],
                                     weight_decay=cfg_merged['weight_decay'])
        scheduler = None

    best_val, patience_ctr, best_state = float('inf'), 0, None

    for epoch in range(1, cfg_merged['max_epochs_s2']+1):
        model.train()
        model.encoder.eval()
        for X, T, _ in train_loader:
            X, T = X.to(device), T.to(device)
            opt.zero_grad()
            with torch.no_grad():
                S, C0 = model.encoder(X)
            loss = diffusion_loss_fn(model.denoiser, C0, S, T,
                                     alphas_bar, ARCH['T_diff'], device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.denoiser.parameters(), cfg_merged['grad_clip'])
            opt.step()
            if scheduler: scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, T, _ in val_loader:
                X, T  = X.to(device), T.to(device)
                S, C0 = model.encoder(X)
                val_loss += diffusion_loss_fn(model.denoiser, C0, S, T,
                                              alphas_bar, ARCH['T_diff'], device).item()
        val_loss /= len(val_loader)

        if val_loss < best_val - 1e-5:
            best_val     = val_loss
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg_merged['patience']:
                print(f"    S2 early stop epoch {epoch}, best val diff={best_val:.4f}")
                break

    model.load_state_dict(best_state)
    print(f"    Stage 2 best val diff loss: {best_val:.4f}")
    return best_val


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dataset, exp_cfg, device):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    all_pred, all_true = [], []

    for X, T, y in loader:
        X, T, y = X.to(device), T.to(device), y.to(device)
        S, C    = model.encoder(X)
        y_hat   = model.outcome(S, C, T)

        if exp_cfg['use_tta'] and exp_cfg['tta_n_views'] > 1:
            preds = [y_hat]
            for _ in range(exp_cfg['tta_n_views']-1):
                X_aug = (X + torch.randn_like(X)*0.01) * (torch.rand_like(X) > 0.10).float()
                Sa, Ca = model.encoder(X_aug)
                preds.append(model.outcome(Sa, Ca, T))
            y_hat = torch.stack(preds).mean(0)

        all_pred.append(y_hat.cpu())
        all_true.append(y.cpu())

    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    rmse = float(np.sqrt(np.mean((pred-true)**2)))
    mae  = float(np.mean(np.abs(pred-true)))
    ss_r = np.sum((pred-true)**2)
    ss_t = np.sum((true-true.mean())**2)
    r2   = float(1 - ss_r/(ss_t+1e-8))
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


@torch.no_grad()
def get_ite_table(model, dataset, therapy_map, baseline_t, device):
    loader  = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    all_ite = []
    for X, T, y in loader:
        X = X.to(device)
        S, C = model.encoder(X)
        all_ite.append(model.ite(S, C).cpu())
    all_ite = torch.cat(all_ite, dim=0).numpy()
    all_ite[:, baseline_t] = 0.0
    inv_map = {v: k for k, v in therapy_map.items()}
    rows = []
    for t_idx in range(ARCH['n_treatments']):
        col = all_ite[:, t_idx]
        rows.append({
            'drug':     inv_map.get(t_idx, f'T{t_idx}'),
            'mean_ite': float(col.mean()),
            'std':      float(col.std()),
        })
    return sorted(rows, key=lambda r: r['mean_ite'])


# ─────────────────────────────────────────────────────────────────────────────
#  RUN ONE EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(name, exp_cfg, train_ds, val_ds, test_ds,
                   therapy_map, baseline_t, device):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  lr={exp_cfg['lr']} | dropout={exp_cfg['dropout']} | "
          f"AdamW+cos={exp_cfg['use_adamw_cosine']} | TTA={exp_cfg['use_tta']}")
    print(f"{'='*60}")

    set_seed(SEED)

    cfg_merged = {**ARCH, **TRAIN_BASE,
                  'lr': exp_cfg['lr'],
                  'dropout': exp_cfg['dropout'],
                  'diff_dropout': exp_cfg['diff_dropout']}

    model = build_model(exp_cfg, device)

    s1_val_mse = train_stage1(model, train_ds, val_ds, baseline_t, cfg_merged, device)
    s2_val_diff = train_stage2(model, train_ds, val_ds, cfg_merged, exp_cfg, device)

    train_metrics = evaluate(model, train_ds, exp_cfg, device)
    val_metrics   = evaluate(model, val_ds,   exp_cfg, device)
    test_metrics  = evaluate(model, test_ds,  exp_cfg, device)
    ite_rows      = get_ite_table(model, test_ds, therapy_map, baseline_t, device)

    print(f"\n  Train  RMSE={train_metrics['RMSE']:.4f}  MAE={train_metrics['MAE']:.4f}  R²={train_metrics['R2']:.4f}")
    print(f"  Val    RMSE={val_metrics['RMSE']:.4f}  MAE={val_metrics['MAE']:.4f}  R²={val_metrics['R2']:.4f}")
    print(f"  Test   RMSE={test_metrics['RMSE']:.4f}  MAE={test_metrics['MAE']:.4f}  R²={test_metrics['R2']:.4f}")
    print(f"\n  ITE ranking (test set):")
    for r in ite_rows:
        tag = ' ← reference' if r['mean_ite'] == 0.0 else ''
        print(f"    {r['drug']:<12}  mean ITE={r['mean_ite']:>+.4f}  std={r['std']:.4f}{tag}")

    return {
        'name':        name,
        's1_val_mse':  s1_val_mse,
        's2_val_diff': s2_val_diff,
        'train':       train_metrics,
        'val':         val_metrics,
        'test':        test_metrics,
        'ite':         ite_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Seed: {SEED}")

    print("\nLoading data...")
    train_df, val_df, test_df, therapy_map, baseline_t = load_data(DATA_PATH, SEED)
    print(f"  Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")
    print(f"  Baseline treatment: {baseline_t} | Map: {therapy_map}")

    train_ds = DepressionDataset(train_df)
    val_ds   = DepressionDataset(val_df)
    test_ds  = DepressionDataset(test_df)

    # ── Run all 5 experiments ───────────────────────────────────────────────
    all_results = []
    for name, exp_cfg in EXPERIMENTS.items():
        result = run_experiment(name, exp_cfg, train_ds, val_ds, test_ds,
                                therapy_map, baseline_t, device)
        all_results.append(result)

    # ── Final comparison table ──────────────────────────────────────────────
    print("\n\n" + "="*90)
    print("  ABLATION SUMMARY TABLE")
    print("="*90)
    print(f"  {'Experiment':<22} {'S1 ValMSE':>10} {'S2 DiffLoss':>12} "
          f"{'Test RMSE':>10} {'Test MAE':>10} {'Test R²':>8} {'Train R²':>9}")
    print("  " + "-"*87)
    for r in all_results:
        print(f"  {r['name']:<22} "
              f"{r['s1_val_mse']:>10.4f} "
              f"{r['s2_val_diff']:>12.4f} "
              f"{r['test']['RMSE']:>10.4f} "
              f"{r['test']['MAE']:>10.4f} "
              f"{r['test']['R2']:>8.4f} "
              f"{r['train']['R2']:>9.4f}")

    print("\n\n  ITE RANKING COMPARISON")
    print("  " + "-"*87)
    header = f"  {'Drug':<12}" + "".join(f"  {r['name']:<18}" for r in all_results)
    print(header)
    inv_map = {v: k for k, v in therapy_map.items()}
    all_drugs = [inv_map[i] for i in range(ARCH['n_treatments'])]
    for drug in all_drugs:
        row = f"  {drug:<12}"
        for r in all_results:
            ite_val = next((x['mean_ite'] for x in r['ite'] if x['drug']==drug), 0.0)
            row += f"  {ite_val:>+.4f}{'':>12}"
        print(row)

    print("\nDone.")


if __name__ == '__main__':
    main()