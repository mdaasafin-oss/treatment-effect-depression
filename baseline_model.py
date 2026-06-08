#Baseline
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


# ─────────────────────────────────────────────────────────────────────────────
#  1. CONFIG  
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    seed = 42
    data_path = 'data_generated.csv'
    train_frac  = 0.70
    val_frac    = 0.15
    test_frac   = 0.15

    input_dim      = 18
    encoder_hidden = [256, 128]
    ds             = 64
    dc             = 64
    n_treatments   = 6
    treatment_emb  = 16
    prop_hidden    = 64
    outcome_hidden = [128, 64]
    ite_hidden     = 64
    dropout        = 0.1

    T_diff       = 100
    beta_start   = 1e-4
    beta_end     = 0.02
    step_emb_dim = 32
    diff_hidden  = [128, 128, 128]

    lr            = 1e-3
    weight_decay  = 1e-5
    batch_size    = 32
    max_epochs_s1 = 100
    max_epochs_s2 = 100
    patience      = 15

    lambda_stab = 1.0
    lambda_prop = 1.0
    lambda_bal  = 0.5
    lambda_ite  = 0.1
    lambda_reg  = 0.01

    vicreg_v = 25.0
    vicreg_c = 1.0

    prop_clip_eps = 0.10
    winsor_pct    = 1.0
    mmd_sigma     = 1.0
    noise_sigma   = 0.01
    feat_drop_p   = 0.10


# ─────────────────────────────────────────────────────────────────────────────
#  2. UTILS
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_alphas_bar(T_diff: int, beta_start: float, beta_end: float, device):
    betas      = torch.linspace(beta_start, beta_end, T_diff, device=device)
    alphas     = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return alphas_bar


# ─────────────────────────────────────────────────────────────────────────────
#  3. DATA
# ─────────────────────────────────────────────────────────────────────────────
HAMD_COLS = [f'HAMD{str(i).zfill(2)}' for i in range(1, 18)]
FEAT_COLS = ['VISIT'] + HAMD_COLS

DROP_COLS = [
    'RAW_ID', 'THERAPY_STATUS', 'PROTOCOL', 'THERPHAS',
    'THERCODE', 'THERAPY1', 'THERAPY2', 'THERCOD1',
    'THERCOD2', 'AGE', 'GENDER', 'ORIGIN', 'GEOCODE',
]


def load_and_preprocess(path: str, seed: int = 42):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    # HAMD values are 1-4 in raw data; shift to 0-3 for modeling
    df[HAMD_COLS] = df[HAMD_COLS] - 1

    agg = {c: 'mean' for c in HAMD_COLS}
    agg['THERAPY'] = 'first'
    df = df.groupby(['UNIQUEID', 'VISIT'], as_index=False).agg(agg)
    df = df.sort_values(['UNIQUEID', 'VISIT']).reset_index(drop=True)

    df['HAMD_TOTAL'] = df[HAMD_COLS].sum(axis=1)
    df['y'] = df.groupby('UNIQUEID')['HAMD_TOTAL'].shift(-1)
    df = df.dropna(subset=['y']).reset_index(drop=True)

    patients = df['UNIQUEID'].unique()
    train_pts, temp_pts = train_test_split(patients, test_size=0.30, random_state=seed)
    val_pts,  test_pts  = train_test_split(temp_pts, test_size=0.50, random_state=seed)

    train_df = df[df['UNIQUEID'].isin(train_pts)].copy()
    val_df   = df[df['UNIQUEID'].isin(val_pts)].copy()
    test_df  = df[df['UNIQUEID'].isin(test_pts)].copy()

    scaler = StandardScaler()
    train_df[FEAT_COLS] = scaler.fit_transform(train_df[FEAT_COLS])
    val_df[FEAT_COLS]   = scaler.transform(val_df[FEAT_COLS])
    test_df[FEAT_COLS]  = scaler.transform(test_df[FEAT_COLS])
    for split in [train_df, val_df, test_df]:
        split[FEAT_COLS] = split[FEAT_COLS].fillna(0)

    therapy_list = sorted(train_df['THERAPY'].unique())
    therapy_map  = {t: i for i, t in enumerate(therapy_list)}
    unk_idx      = len(therapy_map)
    for split in [train_df, val_df, test_df]:
        split['T'] = split['THERAPY'].map(therapy_map).fillna(unk_idx).astype(int)

    baseline_t = int(train_df['T'].value_counts().idxmax())
    return train_df, val_df, test_df, scaler, therapy_map, baseline_t


class DepressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.X = torch.tensor(df[FEAT_COLS].values, dtype=torch.float32)
        self.T = torch.tensor(df['T'].values,        dtype=torch.long)
        self.y = torch.tensor(df['y'].values,        dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  4. MODEL
# ─────────────────────────────────────────────────────────────────────────────
def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.BatchNorm1d(out_dim),
        nn.Dropout(dropout),
    )


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, ds: int, dc: int, dropout: float):
        super().__init__()
        assert hidden_dims[-1] == ds + dc
        self.ds = ds
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers.append(_mlp_block(in_dim, h, dropout))
            in_dim = h
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return h[:, :self.ds], h[:, self.ds:]


class PropensityHead(nn.Module):
    def __init__(self, dc: int, hidden_dim: int, n_treatments: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dc, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_treatments),
        )

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        return self.net(C)


class OutcomeModel(nn.Module):
    def __init__(self, ds: int, dc: int, n_treatments: int,
                 t_emb_dim: int, hidden_dims, dropout: float):
        super().__init__()
        self.t_embed = nn.Embedding(n_treatments + 1, t_emb_dim)
        in_dim, layers = ds + dc + t_emb_dim, []
        for h in hidden_dims:
            layers.append(_mlp_block(in_dim, h, dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, S: torch.Tensor, C: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([S, C, self.t_embed(T)], dim=-1)).squeeze(-1)


class ITEHead(nn.Module):
    def __init__(self, ds: int, dc: int, hidden_dim: int, n_treatments: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ds + dc, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_treatments),
        )

    def forward(self, S: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([S, C], dim=-1))


class DiffusionDenoiser(nn.Module):
    def __init__(self, dc: int, ds: int, n_treatments: int,
                 t_emb_dim: int, step_emb_dim: int, hidden_dims):
        super().__init__()
        self.step_embed = nn.Sequential(
            nn.Linear(1, step_emb_dim),
            nn.SiLU(),
            nn.Linear(step_emb_dim, step_emb_dim),
        )
        self.t_embed = nn.Embedding(n_treatments + 1, t_emb_dim)
        in_dim, layers = dc + ds + step_emb_dim + t_emb_dim, []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, dc))
        self.net = nn.Sequential(*layers)

    def forward(self, C_noisy: torch.Tensor, step: torch.Tensor,
                S: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        step_emb = self.step_embed(step.float().unsqueeze(-1) / 100.0)
        return self.net(torch.cat([C_noisy, S, step_emb, self.t_embed(T)], dim=-1))


class FullModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder    = Encoder(cfg.input_dim, cfg.encoder_hidden,
                                   cfg.ds, cfg.dc, cfg.dropout)
        self.propensity = PropensityHead(cfg.dc, cfg.prop_hidden, cfg.n_treatments)
        self.outcome    = OutcomeModel(cfg.ds, cfg.dc, cfg.n_treatments,
                                        cfg.treatment_emb, cfg.outcome_hidden,
                                        cfg.dropout)
        self.ite        = ITEHead(cfg.ds, cfg.dc, cfg.ite_hidden, cfg.n_treatments)
        self.denoiser   = DiffusionDenoiser(cfg.dc, cfg.ds, cfg.n_treatments,
                                             cfg.treatment_emb, cfg.step_emb_dim,
                                             cfg.diff_hidden)
        self.cfg = cfg


# ─────────────────────────────────────────────────────────────────────────────
#  5. LOSSES
# ─────────────────────────────────────────────────────────────────────────────
def outcome_loss(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)

def stability_loss(S1, S2):
    return F.mse_loss(S1, S2)

def propensity_loss(logits, T):
    return F.cross_entropy(logits, T)

def _rbf_kernel(X, Y, sigma=1.0):
    XX   = (X ** 2).sum(1, keepdim=True)
    YY   = (Y ** 2).sum(1, keepdim=True)
    dist = XX + YY.t() - 2.0 * (X @ Y.t())
    return torch.exp(-dist / (2.0 * sigma ** 2))

def mmd_loss(C, T, n_treatments, sigma=1.0):
    groups = [C[T == t] for t in range(n_treatments)]
    total, count = torch.tensor(0.0, device=C.device), 0
    for i in range(n_treatments):
        for j in range(i + 1, n_treatments):
            Ci, Cj = groups[i], groups[j]
            if len(Ci) < 2 or len(Cj) < 2:
                continue
            kxx = _rbf_kernel(Ci, Ci, sigma).mean()
            kyy = _rbf_kernel(Cj, Cj, sigma).mean()
            kxy = _rbf_kernel(Ci, Cj, sigma).mean()
            total += kxx + kyy - 2.0 * kxy
            count += 1
    return total / max(count, 1)

def _dr_pseudo_single(S, C, T, y, outcome_model, prop_logits,
                       n_treatments, clip_eps, device):
    with torch.no_grad():
        probs = torch.softmax(prop_logits, dim=-1)
        probs = probs.clamp(clip_eps, 1 - clip_eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)  
        y_hat_obs = outcome_model(S, C, T)
        residual  = y - y_hat_obs
        Y_dr = torch.zeros(S.shape[0], n_treatments, device=device)
        for t in range(n_treatments):
            T_cf       = torch.full((S.shape[0],), t, dtype=torch.long, device=device)
            mu_t       = outcome_model(S, C, T_cf)
            ipw        = (T == t).float() / probs[:, t]
            Y_dr[:, t] = mu_t + ipw * residual
    return Y_dr

def compute_dr_pseudo_outcomes_crossfit(S, C, T, y, outcome_model,
                                         propensity_head, baseline_t,
                                         n_treatments, clip_eps=0.10,
                                         winsor_pct=1.0, device='cpu'):
    B, mid = S.shape[0], S.shape[0] // 2
    tau    = torch.zeros(B, n_treatments, device=device)

    S_a, C_a, T_a, y_a = S[:mid], C[:mid], T[:mid], y[:mid]
    S_b, C_b, T_b, y_b = S[mid:], C[mid:], T[mid:], y[mid:]

    outcome_model.eval()
    with torch.no_grad():
        prop_a = propensity_head(C_b)
        tau[:mid] = _dr_pseudo_single(
            S_a, C_a, T_a, y_a, outcome_model, prop_a,
            n_treatments, clip_eps, device)
        prop_b = propensity_head(C_a)
        tau[mid:] = _dr_pseudo_single(
            S_b, C_b, T_b, y_b, outcome_model, prop_b,
            n_treatments, clip_eps, device)
    outcome_model.train()

    tau = tau - tau[:, baseline_t:baseline_t + 1]

    if winsor_pct < 50:
        lo = np.percentile(tau.cpu().numpy(), winsor_pct,       axis=0)
        hi = np.percentile(tau.cpu().numpy(), 100 - winsor_pct, axis=0)
        lo = torch.tensor(lo, device=device, dtype=torch.float32)
        hi = torch.tensor(hi, device=device, dtype=torch.float32)
        tau = tau.clamp(min=lo, max=hi)

    return tau

def ite_loss(ite_pred, tau_target):
    return F.mse_loss(ite_pred, tau_target)

def vicreg_reg(S, gamma=1.0, v_weight=25.0, c_weight=1.0):
    B, d       = S.shape
    S_centered = S - S.mean(dim=0)
    std    = S_centered.std(dim=0)
    v_loss = torch.mean(F.relu(gamma - std))
    cov    = (S_centered.T @ S_centered) / (B - 1)
    mask   = ~torch.eye(d, dtype=torch.bool, device=S.device)
    c_loss = (cov ** 2)[mask].mean()
    return v_weight * v_loss + c_weight * c_loss

def diffusion_loss(denoiser, C0, S, T, T_diff, alphas_bar, device):
    B   = C0.shape[0]
    tau = torch.randint(1, T_diff + 1, (B,), device=device)
    ab    = alphas_bar[tau - 1].unsqueeze(-1)
    eps   = torch.randn_like(C0)
    C_tau = torch.sqrt(ab) * C0 + torch.sqrt(1 - ab) * eps
    return F.mse_loss(denoiser(C_tau, tau, S, T), eps)

def stage1_loss(batch, model, baseline_t, cfg, device):
    X, T, y = [v.to(device) for v in batch]

    def augment(x):
        noise = torch.randn_like(x) * cfg.noise_sigma
        mask  = (torch.rand_like(x) > cfg.feat_drop_p).float()
        return (x + noise) * mask

    X1, X2 = augment(X), augment(X)
    S1, C1 = model.encoder(X1)
    S2, _  = model.encoder(X2)

    y_pred      = model.outcome(S1, C1, T)
    prop_logits = model.propensity(C1)
    ite_pred    = model.ite(S1, C1)

    tau_target = compute_dr_pseudo_outcomes_crossfit(
        S1.detach(), C1.detach(), T, y,
        model.outcome, model.propensity,
        baseline_t, cfg.n_treatments,
        cfg.prop_clip_eps, cfg.winsor_pct, device,
    )

    L_out  = outcome_loss(y_pred, y)
    L_stab = stability_loss(S1, S2)
    L_prop = propensity_loss(prop_logits, T)
    L_bal  = mmd_loss(C1, T, cfg.n_treatments, cfg.mmd_sigma)
    L_ite  = ite_loss(ite_pred, tau_target)
    L_reg  = vicreg_reg(S1, v_weight=cfg.vicreg_v, c_weight=cfg.vicreg_c)

    total = (L_out
             + cfg.lambda_stab * L_stab
             + cfg.lambda_prop * L_prop
             + cfg.lambda_bal  * L_bal
             + cfg.lambda_ite  * L_ite
             + cfg.lambda_reg  * L_reg)

    return total, {
        'out':   L_out.item(),   'stab': L_stab.item(),
        'prop':  L_prop.item(),  'bal':  L_bal.item(),
        'ite':   L_ite.item(),   'reg':  L_reg.item(),
        'total': total.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  6. TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_stage1(model, train_dataset, val_dataset, baseline_t, cfg, device):
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,  batch_size=cfg.batch_size,
                              shuffle=False)

    s1_params = (list(model.encoder.parameters())
               + list(model.propensity.parameters())
               + list(model.outcome.parameters())
               + list(model.ite.parameters()))

    optimiser = torch.optim.Adam(s1_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val_loss, patience_ctr, best_state = float('inf'), 0, None
    history = []

    print("\n Stage 1: Representation + Outcome Learning")
    for epoch in range(1, cfg.max_epochs_s1 + 1):
        model.train()
        train_comp = {k: 0.0 for k in ['out','stab','prop','bal','ite','reg','total']}

        for batch in train_loader:
            optimiser.zero_grad()
            loss, comp = stage1_loss(batch, model, baseline_t, cfg, device)
            loss.backward()
            nn.utils.clip_grad_norm_(s1_params, 5.0)
            optimiser.step()
            for k in comp:
                train_comp[k] += comp[k]

        n = len(train_loader)
        train_comp = {k: v / n for k, v in train_comp.items()}

        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for X, T, y in val_loader:
                X, T, y = X.to(device), T.to(device), y.to(device)
                S, C    = model.encoder(X)
                val_mse += ((model.outcome(S, C, T) - y) ** 2).mean().item()
        val_mse /= len(val_loader)

        history.append({'epoch': epoch, 'train_total': train_comp['total'],
                        'val_mse': val_mse,
                        **{f'train_{k}': v for k, v in train_comp.items()}})

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train_total={train_comp['total']:.4f} | "
                  f"out={train_comp['out']:.4f} | prop={train_comp['prop']:.4f} | "
                  f"val_mse={val_mse:.4f}")

        if val_mse < best_val_loss - 1e-5:
            best_val_loss = val_mse
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"  Early stop at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    print(f"  Stage 1 best val MSE: {best_val_loss:.4f}")
    return history


def train_stage2(model, train_dataset, val_dataset, cfg, device):
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,  batch_size=cfg.batch_size,
                              shuffle=False)

    for p in model.encoder.parameters():
        p.requires_grad = False

    model.encoder.eval()

    optimiser  = torch.optim.Adam(model.denoiser.parameters(),
                                   lr=cfg.lr, weight_decay=cfg.weight_decay)
    alphas_bar = make_alphas_bar(cfg.T_diff, cfg.beta_start, cfg.beta_end, device)
    best_val_loss, patience_ctr, best_state = float('inf'), 0, None
    history = []

    print("\n Stage 2: Diffusion Model (encoder frozen)")
    for epoch in range(1, cfg.max_epochs_s2 + 1):
        model.train()
        # Keep encoder frozen in eval mode 
        model.encoder.eval()

        train_loss = 0.0
        for X, T, _ in train_loader:
            X, T = X.to(device), T.to(device)
            optimiser.zero_grad()
            with torch.no_grad():
                S, C0 = model.encoder(X)
            loss = diffusion_loss(model.denoiser, C0, S, T,
                                   cfg.T_diff, alphas_bar, device)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, T, _ in val_loader:
                X, T = X.to(device), T.to(device)
                S, C0 = model.encoder(X)
                val_loss += diffusion_loss(model.denoiser, C0, S, T,
                                            cfg.T_diff, alphas_bar, device).item()
        val_loss /= len(val_loader)

        history.append({'epoch': epoch, 'train_diff': train_loss, 'val_diff': val_loss})

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train_diff={train_loss:.4f} | val_diff={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"  Early stop at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    print(f"  Stage 2 best val diff loss: {best_val_loss:.4f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
#  7. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_factual(model, dataset, cfg, device):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    all_pred, all_true = [], []
    for X, T, y in loader:
        X, T  = X.to(device), T.to(device)
        S, C  = model.encoder(X)
        y_hat = model.outcome(S, C, T)
        all_pred.append(y_hat.cpu())
        all_true.append(y.cpu())

    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    rmse   = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae    = float(np.mean(np.abs(pred - true)))
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2     = float(1 - ss_res / (ss_tot + 1e-8))
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def print_results(split_name, metrics):
    print(f"\n  {split_name} Results:")
    print(f"    RMSE : {metrics['RMSE']:.4f}")
    print(f"    MAE  : {metrics['MAE']:.4f}")
    print(f"    R²   : {metrics['R2']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  8. TREATMENT EFFECT ANALYSIS 
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def analyze_ite(model, dataset, therapy_map, baseline_t, cfg, device):

    loader  = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    all_ite = []

    for X, T, y in loader:
        X    = X.to(device)
        S, C = model.encoder(X)
        all_ite.append(model.ite(S, C).cpu())

    all_ite  = torch.cat(all_ite, dim=0).numpy()

    # Baseline treatment effect is 0 by definition
    all_ite[:, baseline_t] = 0.0

    inv_map  = {v: k for k, v in therapy_map.items()}
    base_name = inv_map.get(baseline_t, f'T{baseline_t}')

    print(f"\n Treatment Effect Estimation")
    print(f"  Baseline treatment : {base_name}")
    print(f"  Negative ITE = drug reduces HAMD more than baseline = better\n")
    print(f"  {'Treatment':<<14}  {'Mean ITE':>10}  {'Std':>8}  "
          f"{'95% CI':>20}  {'% Patients Benefit':>20}")
    print("  " + "─" * 80)

    rows = []
    for t_idx in range(cfg.n_treatments):
        col      = all_ite[:, t_idx]
        mean_ite = float(col.mean())
        std_ite  = float(col.std())
        ci_lo    = float(np.percentile(col, 2.5))
        ci_hi    = float(np.percentile(col, 97.5))
        pct_b    = float((col < 0).mean() * 100)
        rows.append((inv_map.get(t_idx, f'T{t_idx}'), t_idx,
                     mean_ite, std_ite, ci_lo, ci_hi, pct_b))

    rows.sort(key=lambda r: r[2])

    for drug, t_idx, mean_ite, std_ite, ci_lo, ci_hi, pct_b in rows:
        tag = '  ← baseline (reference)' if t_idx == baseline_t else ''

        if t_idx == baseline_t:
            mean_ite = 0.0
            std_ite  = 0.0
            ci_lo    = 0.0
            ci_hi    = 0.0
            pct_b    = 0.0

        print(f"  {drug:<14}  {mean_ite:>+10.4f}  {std_ite:>8.4f}  "
              f"[{ci_lo:>+7.3f}, {ci_hi:>+7.3f}]  {pct_b:>18.1f}%{tag}")

    best = rows[0]
    print(f"\n  Best treatment : {best[0]}  (mean ITE = {best[2]:+.4f})")


# ─────────────────────────────────────────────────────────────────────────────
#  9. THREE INFERENCE TYPES DEMO 
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def demonstrate_inference_types(model, dataset, therapy_map, baseline_t,
                                 cfg, device, n_patients=3):
    model.eval()
    inv_map    = {v: k for k, v in therapy_map.items()}
    alphas_bar = make_alphas_bar(cfg.T_diff, cfg.beta_start, cfg.beta_end, device)
    betas      = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T_diff, device=device)
    alphas     = 1.0 - betas

    print(f"\n Inference Type Demonstration")
    print(f"  y = next-visit HAMD total score  |  lower = less depressed\n")

    shown, seen = 0, set()

    for idx in range(len(dataset)):
        X, T_obs_t, y_true = dataset[idx]
        t_obs = T_obs_t.item()
        # Counterfactual = baseline treatment if patient is on something else,
        # else the next index (to always show a real switch)
        t_cf = baseline_t if t_obs != baseline_t else (baseline_t + 1) % cfg.n_treatments
        if (t_obs, t_cf) in seen:
            continue
        seen.add((t_obs, t_cf))

        X_t      = X.unsqueeze(0).to(device)
        T_obs_d  = torch.tensor([t_obs], dtype=torch.long, device=device)
        T_cf_d   = torch.tensor([t_cf],  dtype=torch.long, device=device)

        S, C = model.encoder(X_t)

        # (i) Factual
        y_i   = model.outcome(S, C, T_obs_d).item()

        # (ii) CF — same C, new treatment
        y_ii  = model.outcome(S, C, T_cf_d).item()

        # (iii) CF — generate C_cf via reverse DDPM
        C_cf = torch.randn(1, cfg.dc, device=device)
        for t in reversed(range(1, cfg.T_diff + 1)):
            t_ten    = torch.full((1,), t, dtype=torch.long, device=device)
            eps_pred = model.denoiser(C_cf, t_ten, S, T_cf_d)
            a_t      = alphas[t - 1]
            ab_t     = alphas_bar[t - 1]
            ab_prev  = alphas_bar[t - 2] if t > 1 else torch.tensor(1.0, device=device)
            C_cf     = (1.0 / torch.sqrt(a_t)) * \
                       (C_cf - (1 - a_t) / torch.sqrt(1 - ab_t) * eps_pred)
            if t > 1:
                sigma = torch.sqrt((1 - ab_prev) / (1 - ab_t) * betas[t - 1])
                C_cf += sigma * torch.randn_like(C_cf)
        y_iii = model.outcome(S, C_cf, T_cf_d).item()

        obs_name = inv_map.get(t_obs, f'T{t_obs}')
        cf_name  = inv_map.get(t_cf,  f'T{t_cf}')

        print(f"  Patient {shown + 1}  "
              f"[true next-visit HAMD = {y_true.item():.1f}]")
        print(f"  {'Inference type':<<28}  {'Treatment':^14}  {'Predicted y':>12}")
        print(f"  {'─' * 58}")
        print(f"  {'(i)  Factual':<<28}  {obs_name:^14}  {y_i:>12.4f}")
        print(f"  {'(ii) CF — no diffusion':<<28}  {cf_name:^14}  {y_ii:>12.4f}")
        print(f"  {'(iii)CF — with diffusion':<<28}  {cf_name:^14}  {y_iii:>12.4f}")
        delta_no   = y_ii  - y_i
        delta_with = y_iii - y_i
        print(f"  Δ CF (no diffusion)   : {delta_no:>+.4f}")
        print(f"  Δ CF (with diffusion) : {delta_with:>+.4f}\n")

        shown += 1
        if shown >= n_patients:
            break


# ─────────────────────────────────────────────────────────────────────────────
#  10. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.seed)
    print(f"Device: {device} | Seed: {cfg.seed}")

    print("\nLoading and preprocessing data...")
    train_df, val_df, test_df, scaler, therapy_map, baseline_t = \
        load_and_preprocess(cfg.data_path, seed=cfg.seed)

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"  Baseline treatment (most frequent): {baseline_t}")
    print(f"  Treatment map: {therapy_map}")

    train_ds = DepressionDataset(train_df)
    val_ds   = DepressionDataset(val_df)
    test_ds  = DepressionDataset(test_df)

    model    = FullModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    s1_history = train_stage1(model, train_ds, val_ds, baseline_t, cfg, device)
    s2_history = train_stage2(model, train_ds, val_ds, cfg, device)

    # Factual prediction ────────────────────────────────────
    print("\n── Factual Prediction Results (Objective i) ──")
    for name, ds in [('Train', train_ds), ('Val', val_ds), ('Test', test_ds)]:
        print_results(name, evaluate_factual(model, ds, cfg, device))

    # Treatment effect estimation 
    analyze_ite(model, test_ds, therapy_map, baseline_t, cfg, device)

    # All three inference types 
    demonstrate_inference_types(model, test_ds, therapy_map, baseline_t,
                                cfg, device, n_patients=3)

    torch.save({
        'model_state': model.state_dict(),
        'therapy_map': therapy_map,
        'baseline_t':  baseline_t,
        's1_history':  s1_history,
        's2_history':  s2_history,
    }, 'baseline_model.pt')
    print("Model saved → baseline_model.pt")


if __name__ == '__main__':
    main()
