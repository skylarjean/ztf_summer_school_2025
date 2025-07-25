# ──────────────────────────────────────────────────────────────────────────────
#        Multimodal Late-Fusion Transient Classifier  (gated fusion)           │
# ──────────────────────────────────────────────────────────────────────────────
import os, math, argparse, warnings, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, average_precision_score,
                             confusion_matrix, roc_curve, auc, precision_recall_curve,
                             top_k_accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
wandb.require("service")
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# ────────────────────────────── CONFIGURATION ────────────────────────────────
class CFG:
    # data paths
    csv_path      = ".../data_train_multi_10.csv"
    root_path     = ".../data_train_multi/day10"
    test_csv_path = ".../data_test_multi_10.csv"
    test_root     = ".../data_test_multi/day10"

    # experiment
    project       = "multimodal_late_fusion_gated"
    wandb_mode    = "disabled"         # change to 'disabled' for offline

    # optimisation
    batch_size    = 128
    epochs        = 150
    lr            = 1e-6
    weight_decay  = 1e-2
    patience      = 50
    num_workers   = 10
    gamma         = 2.0              # focal-loss γ
    seed          = 42

    # ── photometry Transformer
    d_model       = 128
    num_layers    = 8
    num_heads     = 4
    dropout       = 0.3
    max_len       = 500

    # ── spectra CNN
    grid_size           = 59
    conv1_filters       = 32
    conv1_kernel        = 7
    conv2_filters       = 64
    conv2_kernel        = 7
    pool_size           = 2
    fc_units_spec       = 2048
    fc_dropout          = 0.3

    # ── astroMiNN (image + metadata)  ← new tunables
    astro_input_dim           = 9
    astro_hidden_dim          = 128     # tower hidden dim (was 128)
    astro_proj_dim            = 256     # tower projection dim (was 256)
    astro_num_experts         = 10      # classification experts
    astro_num_fusion_experts  = 5       # metadata/image fusion experts

cfg = CFG()

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ─────────────────────────────── TAXONOMY ─────────────────────────────────────
BROAD_CLASSES = ["SNI", "SNII", "Cataclysmic", "AGN", "TDE"]
NUM_CLASSES   = len(BROAD_CLASSES)
ORIG2BROAD = {
    "SN Ia":"SNI","SN Ib":"SNI","SN Ic":"SNI",
    "SN II":"SNII","SN IIP":"SNII","SN IIn":"SNII","SN IIb":"SNII",
    "Cataclysmic":"Cataclysmic","AGN":"AGN","Tidal Disruption Event":"TDE"
}
BROAD2ID = {c:i for i,c in enumerate(BROAD_CLASSES)}

# ─────────────────── NORMALISATION STATS FOR METADATA ────────────────────────
METADATA_STATS = {
    'sgscore1': {'mean': 0.236, 'std': 0.266},
    'sgscore2': {'mean': 0.401, 'std': 0.328},
    'distpsnr1': {'mean': 3.151, 'std': 3.757},
    'distpsnr2': {'mean': 9.269, 'std': 6.323},
    'nmtchps': {'mean': 9.231, 'std': 8.089},
    'sharpnr': {'mean': 0.253, 'std': 0.256},
    'scorr': {'mean': 22.089, 'std': 16.757},
    'ra': {'mean': 0, 'std': 1},
    'dec': {'mean': 0, 'std': 1}
}

# ───────────────────────────── DATASET ────────────────────────────────────────
class MultiModalDataset(Dataset):
    """
    Returns:
        phot       – [T,4]
        spec_img   – [1,H,W]
        meta_vec   – [9]
        image      – [3,H,W]
        label      – int
    """
    def __init__(self, df: pd.DataFrame, root: Path,
                 phot_mean=None, phot_std=None, spec_scaler=None,
                 grid_size=cfg.grid_size, transform=None):
        self.df   = df.reset_index(drop=True)
        self.root = Path(root)
        self.phot_mean = phot_mean
        self.phot_std  = phot_std
        self.spec_scaler = spec_scaler
        self.grid_size   = grid_size
        self.transform   = transform

    def __len__(self): return len(self.df)

    def _load_sample(self, row):
        sample = np.load(self.root / row["file"], allow_pickle=True).item()
        phot   = torch.tensor(sample["photometry"], dtype=torch.float32)
        spec   = np.nan_to_num(sample["spectra"][0])

        # metadata vector
        m = sample['metadata']
        meta_vec = torch.tensor([
            (m['sgscore1']  - METADATA_STATS['sgscore1']['mean']) / METADATA_STATS['sgscore1']['std'],
            (m['sgscore2']  - METADATA_STATS['sgscore2']['mean']) / METADATA_STATS['sgscore2']['std'],
            (m['distpsnr1'] - METADATA_STATS['distpsnr1']['mean']) / METADATA_STATS['distpsnr1']['std'],
            (m['distpsnr2'] - METADATA_STATS['distpsnr2']['mean']) / METADATA_STATS['distpsnr2']['std'],
            (m['nmtchps']   - METADATA_STATS['nmtchps']['mean'])  / METADATA_STATS['nmtchps']['std'],
            (m['sharpnr']   - METADATA_STATS['sharpnr']['mean'])  / METADATA_STATS['sharpnr']['std'],
            (m['scorr']     - METADATA_STATS['scorr']['mean'])    / METADATA_STATS['scorr']['std'],
            (m['ra'] / 360.0),
            ((m['dec'] + 90.0) / 180.0)
        ], dtype=torch.float32)

        # image
        img = torch.tensor(sample['images'], dtype=torch.float32).permute(2,0,1) / 255.0
        if self.transform: img = self.transform(img)
        return phot, spec, meta_vec, img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        phot, spec, meta_vec, img = self._load_sample(row)

        # spectra → square grid
        spec_std = self.spec_scaler.transform([spec])[0]
        needed   = self.grid_size**2
        if spec_std.shape[0] < needed:
            spec_std = np.pad(spec_std, (0, needed-spec_std.shape[0]))
        elif spec_std.shape[0] > needed:
            spec_std = spec_std[:needed]
        spec_img = torch.tensor(spec_std.reshape(self.grid_size, self.grid_size),
                                dtype=torch.float32).unsqueeze(0)

        label = BROAD2ID[row["type"]]
        return phot, spec_img, meta_vec, img, label

def compute_phot_mean_std(ds):
    sums, sqs, n = torch.zeros(4), torch.zeros(4), 0
    for phot, *_ in ds:
        sums += phot.sum(0)
        sqs  += (phot**2).sum(0)
        n    += phot.shape[0]
    mean = (sums / n)[None,None,:]
    std  = torch.sqrt(sqs/n - mean.squeeze()**2)[None,None,:]
    return mean, std

def build_collate(mean, std):
    def collate(batch):
        phot_seqs, spec_imgs, meta_vecs, img_tensors, labels = zip(*batch)

        spec_batch = torch.stack(spec_imgs)
        meta_batch = torch.stack(meta_vecs)
        img_batch  = torch.stack(img_tensors)

        lengths = [p.shape[0] for p in phot_seqs]
        padded  = pad_sequence(phot_seqs, batch_first=True)
        mask    = torch.stack([
            torch.cat([torch.zeros(l), torch.ones(padded.shape[1]-l)])
            for l in lengths
        ]).bool()
        normed  = (padded - mean) / (std + 1e-8)
        return normed, mask, spec_batch, meta_batch, img_batch, torch.tensor(labels)
    return collate

# ─────────────────────────────── MODELS ───────────────────────────────────────
# === photometry Transformer ===================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1e4)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:,:x.size(1)]

class PhotometryEncoder(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, cfg.d_model)
        self.cls   = nn.Parameter(torch.zeros(1,1,cfg.d_model))
        self.pos   = PositionalEncoding(cfg.d_model, cfg.max_len+1)
        enc_layer  = nn.TransformerEncoderLayer(cfg.d_model, cfg.num_heads,
                                                cfg.d_model*4, cfg.dropout, batch_first=True)
        self.enc   = nn.TransformerEncoder(enc_layer, cfg.num_layers)
        self.norm  = nn.LayerNorm(cfg.d_model)
    def forward(self, x, pad_mask):
        B = x.size(0)
        tok = self.cls.expand(B,-1,-1)
        x   = torch.cat([tok, self.embed(x)],1)
        pad = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=x.device), pad_mask],1)
        h   = self.enc(self.pos(x), src_key_padding_mask=pad)
        return self.norm(h[:,0])

# === spectra CNN (DASH-style) ================================================
class SpectraEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, cfg.conv1_filters, cfg.conv1_kernel, padding=cfg.conv1_kernel//2)
        self.bn1   = nn.BatchNorm2d(cfg.conv1_filters)
        self.pool1 = nn.MaxPool2d(cfg.pool_size)
        self.conv2 = nn.Conv2d(cfg.conv1_filters, cfg.conv2_filters, cfg.conv2_kernel, padding=cfg.conv2_kernel//2)
        self.bn2   = nn.BatchNorm2d(cfg.conv2_filters)
        self.pool2 = nn.MaxPool2d(cfg.pool_size)
        out_size   = cfg.grid_size // (cfg.pool_size * cfg.pool_size)
        self.flat  = nn.Flatten()
        self.fc1   = nn.Linear(cfg.conv2_filters*out_size*out_size, cfg.fc_units_spec)
        self.drop  = nn.Dropout(cfg.fc_dropout)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = self.flat(x)
        return self.drop(F.relu(self.fc1(x)))

# === astroMiNN (image + metadata) ============================================
from torchvision.ops import SqueezeExcitation       # noqa
from torchvision.models import DenseNet             # noqa
from CNN import CoordCNNJointTower           # assumes same dependency

class FeatureInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj  = nn.Linear(dim, dim*2)
        self.gate  = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(self.proj(x))

class astroMiNN(nn.Module):
    """
    Image + metadata tower with fusion/classification MoEs.
    All tunables come from CFG, keeping original defaults.
    """
    def __init__(self):
        super().__init__()
        H, P = cfg.astro_hidden_dim, cfg.astro_proj_dim  # 128, 256 by default

        self.psf_tower = nn.Sequential(
            nn.Linear(2, H), FeatureInteraction(H), nn.GELU(),
            nn.LayerNorm(H), nn.Linear(H, P), nn.SiLU())

        self.spatial_tower = nn.Sequential(
            nn.Linear(3, H), FeatureInteraction(H), nn.GELU(),
            nn.LayerNorm(H), nn.Linear(H, P), nn.SiLU())

        self.nst1_tower = nn.Sequential(
            nn.Linear(2, H), FeatureInteraction(H), nn.GELU(),
            nn.LayerNorm(H), nn.Linear(H, P), nn.SiLU())

        self.nst2_tower = nn.Sequential(
            nn.Linear(2, H), FeatureInteraction(H), nn.GELU(),
            nn.LayerNorm(H), nn.Linear(H, P), nn.SiLU())

        self.coord_cnn_tower = CoordCNNJointTower()  # from your codebase

        # fusion MoE
        self.fusion_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(P*5, 512), nn.LayerNorm(512),
                          nn.GELU(), nn.Linear(512, P))
            for _ in range(cfg.astro_num_fusion_experts)   # 5 default
        ])
        self.fusion_router = nn.Linear(P*5, cfg.astro_num_fusion_experts)
        self.proj_fuse     = nn.Sequential(nn.Linear(P, P), nn.LayerNorm(P), nn.GELU())

        # classification MoE
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(P, P//2), nn.LayerNorm(P//2),
                          nn.GELU(), nn.Dropout(.1),
                          nn.Linear(P//2, NUM_CLASSES))
            for _ in range(cfg.astro_num_experts)          # 10 default
        ])
        self.router = nn.Linear(P, cfg.astro_num_experts)

    def forward(self, x_meta, image):
        psf_feats = self.psf_tower(x_meta[:, [5,6]])            # sharpnr, scorr
        spatial   = self.spatial_tower(x_meta[:, [2,3,4]])      # distpsnr1,2,nmtchps
        nsta      = self.nst1_tower(x_meta[:, [0,2]])           # sgscore1, distpsnr1
        nstb      = self.nst2_tower(x_meta[:, [1,3]])           # sgscore2, distpsnr2
        joint     = self.coord_cnn_tower(x_meta[:, [7,8]], image)

        all_feats = torch.cat([nsta, nstb, spatial, psf_feats, joint], 1)
        fusion_w  = F.softmax(self.fusion_router(all_feats), 1)
        expert_out = torch.stack([e(all_feats) for e in self.fusion_experts], 1)
        fused     = torch.einsum('be,bej->bj', fusion_w, expert_out)
        fused     = self.proj_fuse(fused)

        router_logits = self.router(fused)
        expert_w      = F.softmax(router_logits, 1)
        expert_logits = torch.stack([e(fused) for e in self.experts], 2)
        return torch.einsum('be,bce->bc', expert_w, expert_logits)  # logits

# === overall multimodal model with gated late-fusion ==========================
class MultiModalClassifier(nn.Module):
    """
    Branch-1 : photometry Transformer
    Branch-2 : spectra CNN
    Branch-3 : astroMiNN (image + metadata)
    Fusion    : sample-adaptive gating network over per-branch logits.
    """
    def __init__(self):
        super().__init__()
        # phot + spec
        self.phot = PhotometryEncoder()
        self.spec = SpectraEncoder()
        fusion_dim = cfg.d_model + cfg.fc_units_spec
        self.head  = nn.Sequential(nn.Linear(fusion_dim, 512),
                                   nn.ReLU(), nn.Dropout(0.3),
                                   nn.Linear(512, NUM_CLASSES))

        # astroMiNN
        self.astro = astroMiNN()

        # gated fusion (takes concatenated logits → weights)
        self.gate = nn.Sequential(
            nn.Linear(NUM_CLASSES*2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)            # weights for the 2 branches
        )

    def forward(self, phot, mask, spec, meta, img):
        h_phot = self.phot(phot, mask)
        h_spec = self.spec(spec)
        logits_ps = self.head(torch.cat([h_phot, h_spec], 1))   # phot+spec logits
        logits_im = self.astro(meta, img)                       # image+meta logits

        stacked = torch.cat([logits_ps, logits_im], 1)          # [B, 2*C]
        w = torch.softmax(self.gate(stacked), 1)                # [B,2]
        final_logits = w[:,0:1]*logits_ps + w[:,1:2]*logits_im
        return final_logits

# ─────────────────────────────── LOSSES ───────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=cfg.gamma, alpha=None):
        super().__init__(); self.g, self.a = gamma, alpha
    def forward(self, logits, target):
        logp = F.log_softmax(logits,1); p = logp.exp()
        idx  = torch.arange(logits.size(0), device=logits.device)
        logp_t, p_t = logp[idx,target], p[idx,target]
        loss = -((1-p_t)**self.g) * logp_t
        if self.a is not None: loss *= self.a[target]
        return loss.mean()

# ─────────────────────────── DATA PREPARATION ────────────────────────────────
full_df = pd.read_csv(cfg.csv_path)
full_df["type"] = full_df["type"].map(ORIG2BROAD)



# ↓↓↓ NEW: strip rows whose .npy is missing ↓↓↓
exist_mask = full_df["file"].apply(lambda f: (Path(cfg.root_path) / f).exists())
missing_rows = (~exist_mask).sum()
if missing_rows:
    warnings.warn(f"⚠️  dropping {missing_rows} samples with missing .npy files")
full_df = full_df[exist_mask].reset_index(drop=True)





train_df, val_df = np.split(full_df.sample(frac=1,random_state=cfg.seed),
                            [int(0.8*len(full_df))])




# spectra scaler
spec_scaler = StandardScaler()
for f in train_df["file"]:
    arr = np.nan_to_num(np.load(Path(cfg.root_path)/f, allow_pickle=True).item()["spectra"][0])
    spec_scaler.partial_fit(arr.reshape(1,-1))

# datasets / loaders
train_ds = MultiModalDataset(train_df, cfg.root_path, spec_scaler=spec_scaler,
                             grid_size=cfg.grid_size)
phot_mean, phot_std = compute_phot_mean_std(train_ds)
train_ds.phot_mean, train_ds.phot_std = phot_mean, phot_std
val_ds   = MultiModalDataset(val_df, cfg.root_path, phot_mean, phot_std,
                             spec_scaler, cfg.grid_size)

collate = build_collate(phot_mean, phot_std)
train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.num_workers, collate_fn=collate, drop_last=True)
val_ld   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                      num_workers=cfg.num_workers, collate_fn=collate)

# class imbalance → α for focal loss
labels_train = torch.tensor([train_ds[i][-1] for i in range(len(train_ds))])
counts = torch.bincount(labels_train, minlength=NUM_CLASSES).float()
alpha  = (1.0 / torch.sqrt(counts)).to(device)

# ───────────────────────────── TRAINING LOOP ──────────────────────────────────
wandb.init(project=cfg.project, mode=cfg.wandb_mode, config=vars(cfg))
model = MultiModalClassifier().to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10,T_mult=2,eta_min=1e-6)
crit  = FocalLoss(cfg.gamma, alpha)

best, no_imp = 0.0, 0
ckpt_path = Path(wandb.run.dir) / "best.pt"

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tl, correct, N = 0., 0, 0
    all_prob, all_lab = [], []
    with torch.set_grad_enabled(train):
        for phot, mask, spec, meta, img, y in loader:
            phot, mask, spec = phot.to(device), mask.to(device), spec.to(device)
            meta, img, y     = meta.to(device), img.to(device), y.to(device)

            logits = model(phot, mask, spec, meta, img)
            loss   = crit(logits, y)

            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); sched.step()

            tl += loss.item()*y.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            N += y.size(0)
            all_prob.append(F.softmax(logits,1).detach().cpu())
            all_lab.append(y.cpu())

    probs = torch.cat(all_prob); labs = torch.cat(all_lab)
    acc   = correct / N
    auroc = roc_auc_score(labs, probs, multi_class="ovr", average="macro")
    return tl/N, acc, auroc, labs.numpy(), probs.numpy()

for ep in range(1, cfg.epochs+1):
    tr_loss, tr_acc, tr_auc, *_ = run_epoch(train_ld, True)
    va_loss, va_acc, va_auc, yt, yp = run_epoch(val_ld, False)

    y_onehot = label_binarize(yt, classes=np.arange(NUM_CLASSES))
    va_auprc = average_precision_score(y_onehot, yp, average="macro")

    wandb.log({"epoch":ep, "train/loss":tr_loss, "train/acc":tr_acc,
               "val/loss":va_loss, "val/acc":va_acc,
               "val/auroc_macro":va_auc, "val/auprc_macro":va_auprc,
               "lr":sched.get_last_lr()[0]})

    if va_auprc > best:
        best, no_imp = va_auprc, 0
        torch.save(model.state_dict(), ckpt_path)
        wandb.save(str(ckpt_path))
    else:
        no_imp += 1
        if no_imp >= cfg.patience:
            print("Early stopping"); break
    print(f"[{ep:03d}] train_acc={tr_acc:.3f} val_acc={va_acc:.3f} val_AUPRC={va_auprc:.3f}")

# ─────────────────────────── EVALUATION & PLOTS ───────────────────────────────
def evaluate(loader, split):
    model.load_state_dict(torch.load(ckpt_path))
    loss, acc, auc_macro, y, p = run_epoch(loader, False)
 # YOUR CODE HERE :) ...
val_metrics = evaluate(val_ld, "val")

# test set
#test_df = pd.read_csv(cfg.test_csv_path)
#test_df["type"] = test_df["type"].map(ORIG2BROAD)
#test_ds = MultiModalDataset(test_df, cfg.test_root, phot_mean, phot_std,
#                            spec_scaler, cfg.grid_size)
#test_ld = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
#                     num_workers=cfg.num_workers, collate_fn=collate)
#test_metrics = evaluate(test_ld, "test")

#wandb.finish()

