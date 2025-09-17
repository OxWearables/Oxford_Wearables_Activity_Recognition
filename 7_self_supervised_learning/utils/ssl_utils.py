"""
Teaching template for SSL on wearable (accelerometer) windows.

Design goals:
- Students only edit:
  - augmentations in `Augmenter`
  - dataset return values (labels / pairs)
  - the inner loss functions (Aug-Rec CE; NT-Xent)
- Standard backbone + small heads; same training loop style for both tasks
- Minimal dependency footprint; NO Lightning

Structure (single file for teaching; can be split later):
- Config
- Augmentations
- Datasets: BaseWearableDataset, AugRecDataset, ContrastiveDataset
- Models: Backbone1D, ProjectionHead, AugRecHead, DownstreamHead
- SSL losses: augrec_bce_with_logits, nt_xent_loss
- Trainer: generic loop + EarlyStopping
- Finetune + Eval on CAPTURE24

Notes:
- Marked TODOs are student-facing. Keep them small & local.
- Use CPU by default; CUDA picked up if available.
"""

from __future__ import annotations
from tqdm import tqdm
import os
import shutil, zipfile, urllib.request
import math
import random
from dataclasses import dataclass, fields
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Seed
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
#  Data IO: symlink-from-/srv OR download locally, then load & split
# ------------------------e

class Capture24DataManager:
    """
    Single entry point for Capture-24 data.

    Example
    -------
    dm = Capture24DataManager(srv_root="/srv", local_root=".")
    dm.prepare()  # symlink from /srv or download locally (processed_data + annotation dict)
    X, Y, T, pid = dm.load_arrays(max_size=30_000, map_willetts=True)
    splits = dm.train_val_test_split(test_size=0.2, val_size=0.125, seed=42)
    (x_tr, y_tr, pid_tr, x_val, y_val, pid_val, x_te, y_te, pid_te, le) = splits
    """

    PROC_URL = "https://wearables-files.ndph.ox.ac.uk/files/processed_data.zip"
    CAP_URL  = (
        "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/"
        "download_file?file_format=&safe_filename=capture24.zip&type_of_work=Dataset"
    )

    def __init__(self, srv_root: str = "/srv", local_root: str = ".."):
        self.srv_root = srv_root
        self.local_root = local_root
        self.cap_dir  = os.path.join(local_root, "capture24")
        self.proc_dir = os.path.join(local_root, "processed_data")
        # populated by load_arrays()
        self.X = self.Y = self.T = self.pid = None

    # ---- private helpers ----
    @staticmethod
    def _symlink(src: str, dst: str):
        if os.path.exists(dst):
            return
        try:
            os.symlink(src, dst)
            print(f"Linked {dst} -> {src}")
        except (OSError, NotImplementedError) as e:
            print(f"Symlink failed ({e}); copying (may take a while)...")
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    @staticmethod
    def _download(url: str, out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with urllib.request.urlopen(url) as f_src, open(out_path, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)

    # ---- public API ----
    def prepare(self):
        """
        Ensure ./processed_data and ./capture24 exist.
        - If /srv has them, create local symlinks.
        - Otherwise download processed_data.zip and capture24.zip (for annotation dict).
        """
        srv_cap  = os.path.join(self.srv_root, "capture24")
        srv_proc = os.path.join(self.srv_root, "processed_data")

        # Prefer the classroom mount when available
        if os.path.exists(srv_cap):
            self._symlink(srv_cap, self.cap_dir)
        if os.path.exists(srv_proc):
            self._symlink(srv_proc, self.proc_dir)

        # processed_data fallback
        if not os.path.exists(self.proc_dir):
            zip_path = os.path.join(self.local_root, "processed_data.zip")
            print("Downloading processed_data.zip ...")
            self._download(self.PROC_URL, zip_path)
            print("Unzipping processed_data.zip ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.local_root)
            print("Processed data ready.")
        else:
            print("Using existing processed_data/ directory.")

        # capture24 fallback (for annotation dictionary)
        dict_path = os.path.join(self.cap_dir, "annotation-label-dictionary.csv")
        if not os.path.exists(dict_path):
            zip_path = os.path.join(self.local_root, "capture24.zip")
            print("Downloading capture24.zip (annotation dictionary)...")
            self._download(self.CAP_URL, zip_path)
            print("Unzipping capture24.zip ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.local_root)
            print("Capture24 (annotation dictionary) ready.")
        else:
            print("Using existing capture24/ directory.")


    def train_val_test_split(
            self,
            prop: int | None = None,
            segment_size: int | None = 900,
            map_willetts: bool = True,
            test_size=0.2,
            val_size=0.125,
            seed=42):
        """
        Group-wise split by participant ID.
        Returns: (X_tr, y_tr, pid_tr, X_val, y_val, pid_val, X_te, y_te, pid_te, label_encoder_or_None)
        """
        X   = np.load(os.path.join(self.proc_dir, "X.npy"), mmap_mode="r")
        Y   = np.load(os.path.join(self.proc_dir, "Y.npy"))
        T   = np.load(os.path.join(self.proc_dir, "T.npy"))
        pid = np.load(os.path.join(self.proc_dir, "pid.npy"))

        if prop is not None:
            max_size = int(len(X)*prop)
            X, Y, T, pid = X[:max_size], Y[:max_size], T[:max_size], pid[:max_size]

        if map_willetts:
            csv_path = os.path.join(self.cap_dir, "annotation-label-dictionary.csv")
            if os.path.exists(csv_path):
                import pandas as pd
                anno_label_dict = pd.read_csv(csv_path, index_col="annotation", dtype="string")
                try:
                    Y = anno_label_dict.loc[Y, "label:Willetts2018"].to_numpy()
                except Exception as e:
                    print(f"Label mapping failed ({e}); keeping original Y.")
            else:
                print("annotation-label-dictionary.csv not found; keeping original Y.")

        # If last axis looks like channels, transpose to (N,C,L)
        if X.shape[-1] in (1, 3):
            X = np.transpose(X, (0, 2, 1))

        # Optionally resample X data
        if segment_size is not None:
            length_orig = X.shape[-1]
            t_orig = np.linspace(0, 1, length_orig, endpoint=True)
            t_new = np.linspace(0, 1, segment_size, endpoint=True)
            X = interp1d(t_orig, X, kind="linear", axis=-1, assume_sorted=True)(
                t_new
            )
        
        # Get references to the unsplit data
        self.X, self.Y, self.T, self.pid = X, Y, T, pid

        le = None
        # encode if string/object labels
        if hasattr(Y, "dtype") and Y.dtype.kind in {"U", "S", "O"}:
            le = LabelEncoder(); le.fit(np.unique(Y)); y = le.transform(Y)
        else:
            y = Y.astype(int)

        gss = GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(X, y, groups=pid)
        train_idx, test_idx = next(gss)
        X_tr_full, X_te = X[train_idx], X[test_idx]
        y_tr_full, y_te = y[train_idx], y[test_idx]
        pid_tr_full, pid_te = pid[train_idx], pid[test_idx]

        gss2 = GroupShuffleSplit(1, test_size=val_size, random_state=seed+1).split(
            X_tr_full, y_tr_full, groups=pid_tr_full
        )
        tr_idx, val_idx = next(gss2)
        X_tr, X_val = X_tr_full[tr_idx], X_tr_full[val_idx]
        y_tr, y_val = y_tr_full[tr_idx], y_tr_full[val_idx]
        pid_tr, pid_val = pid_tr_full[tr_idx], pid_tr_full[val_idx]

        return (X_tr, y_tr, pid_tr, X_val, y_val, pid_val, X_te, y_te, pid_te, le)


# -------------------------
# Augmentations - appears in the notebook!
# -------------------------
@dataclass
class AugmentConfig:
    jitter: float = 0.5
    scaling: float = 0.5
    time_flip: float = 0.5
    axis_swap: float = 0.2
    time_mask: float = 0.3


class Augmenter:
    """
    Composable time-series augs tailored to wrist accelerometer windows.
    """
    def __init__(self, cfg: AugmentConfig | None = None):
        # if None, fall back to defaults
        self.cfg = cfg or AugmentConfig()

    @classmethod
    def available_ops(cls) -> list[str]:
        """Ops = config fields that have a same-named augmentation method."""
        names = [f.name for f in fields(AugmentConfig)]  # preserves declaration order
        return [n for n in names if hasattr(cls, n) and callable(getattr(cls, n))]
    
    def probs(self) -> dict[str, float]:
        """Current op -> probability mapping from the config."""
        return {n: getattr(self.cfg, n) for n in self.available_ops()}

    # ---- primitive ops: (C, L) -> (C, L) ----
    @staticmethod
    def jitter(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
        # add small Gaussian noise
        return x + torch.randn_like(x) * sigma

    @staticmethod
    def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        # per-sample scalar scale ~ N(1, sigma^2)
        s = torch.randn((), device=x.device) * sigma + 1.0  # shape ()
        return x * s

    @staticmethod
    def time_flip(x: torch.Tensor) -> torch.Tensor:
        # reverse along time axis
        return torch.flip(x, dims=[-1])

    @staticmethod
    def axis_swap(x: torch.Tensor) -> torch.Tensor:
        # swap y and z (requires >=3 channels); no-op otherwise
        if x.size(0) >= 3:
            return x[[0, 2, 1], :]
        return x

    @staticmethod
    def time_mask(x: torch.Tensor, max_frac: float = 0.1) -> torch.Tensor:
        # zero a contiguous span of the series
        L = x.size(-1)
        w = max(1, int(L * max_frac))
        start = random.randint(0, L - w)
        y = x.clone()
        y[:, start:start + w] = 0
        return y

    # --- pipelines ---
    def view(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stochastic augmentation pipeline.
        """
        if random.random() < self.cfg.jitter:
            x = self.jitter(x)
        if random.random() < self.cfg.scaling:
            x = self.scaling(x)
        if random.random() < self.cfg.axis_swap:
            x = self.axis_swap(x)
        if random.random() < self.cfg.time_mask:
            x = self.time_mask(x)
        if random.random() < self.cfg.time_flip:
            x = self.time_flip(x)
        return x

    def two_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.view(x), self.view(x)


# -------------------------
# Visualisation
# -------------------------

def visualize_segment(
        x: np.ndarray | torch.Tensor,
        fs: float | None = 30.0,
        title: str | None = None,
        figsize: Tuple[int] | None = (4,3),
    ):
    """
    Plot an accelerometer segment shaped (C, L).
    - x: (C, L) numpy array or torch tensor
    - fs: sampling rate in Hz (set None to use sample index on x-axis)
    """
    # to numpy, check shape
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected (C, L), got {x.shape}")
    C, L = x.shape
    if C not in (1, 3):
        print(f"Warning: unusual channel count C={C}. Plotting first 3 if present.")
    ch_names = ["x", "y", "z"]

    # time axis
    if fs and fs > 0:
        t = np.arange(L) / fs
        xlab = "time (s)"
    else:
        t = np.arange(L)
        xlab = "sample"

    # set up subplots
    rows = min(C, 3)
    fig, ax = plt.subplots(figsize=figsize)

    for i, name in enumerate(ch_names[:rows]):
        ax.plot(t, x[i], label=name)
    
    ax.legend()
    ax.set_xlabel(xlab)
    ax.set_ylabel("Acc.")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

    return fig, ax


# -------------------------
# Datasets - shown in the notebook
# -------------------------

class BaseWearableDataset(Dataset):
    """
    X is (N, C, L). If y is provided, returns (x, y) for supervised finetuning.
    Otherwise returns x only (useful for SSL or inference).
    Optional light augmentation can be applied via `aug_prob`.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        augmenter: Optional[Augmenter] = None,
    ):
        assert X.ndim == 3, f"Expected (N,C,L), got {X.shape}"
        self.X = X
        self.y = y
        self.aug = augmenter

    def __len__(self):
        return len(self.X)

    def _get_x(self, idx: int) -> torch.Tensor:
        # (C, L) float32
        return torch.from_numpy(self.X[idx]).float()
    
    def _get_y(self, idx: int) -> torch.Tensor | None:
        # long
        return torch.tensor(self.y[idx], dtype=torch.long)

    def __getitem__(self, idx: int):
        x = self._get_x(idx)  # (C, L)

        # optional light augmentation for supervised finetuning
        if self.aug is not None:
            x = self.aug.view(x)  # expects (C,L) -> (C,L)

        if self.y is None:
            return x

        # robust label -> long tensor
        y = self._get_y(idx)
        return x, y

class AugRecDataset(BaseWearableDataset):
    def __init__(self, *args, multi_label: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_label = multi_label
        self.ops = self.aug.available_ops()          # dynamic, ordered
        self._op_probs = self.aug.probs()            # dict for quick lookup

    def __getitem__(self, idx):
        x = self._get_x(idx)                         # (C, L)
        labels = torch.zeros(len(self.ops), dtype=torch.float32)

        for k, op in enumerate(self.ops):
            p = self._op_probs[op]
            if random.random() < p:
                labels[k] = 1.0
                x = getattr(self.aug, op)(x)         # call op by name

        if not self.multi_label:
            labels = labels.max().unsqueeze(0)       # binary: any-aug

        return x, labels

class ContrastiveDataset(BaseWearableDataset):
    def __getitem__(self, idx):
        x = self._get_x(idx)              # (C, L)
        v1, v2 = self.aug.two_views(x)    # both (C, L)
        return v1, v2

# -------------------------
# Models
# -------------------------

@dataclass
class HubConfig:
    repo: str = "OxWearables/ssl-wearables"
    entry: str = "harnet30"
    class_num: int = 6
    pretrained: bool = True
    commit: str | None = "150550ea5d41800229c95e36f88f5bf0d2e7cf04"
    trust_repo: bool = True
    weights_only: bool = False
    force_reload: bool = False
    skip_validation: bool = True  # avoids import-time validation

@dataclass
class ModelConfig:
    in_channels: int = 3
    input_len: int = 900
    proj_dim: int = 128
    num_classes: int = 4
    k_labels: int = 5
    freeze_backbone: bool = False

@dataclass
class SSLConfig:
    hub: HubConfig = HubConfig()
    model: ModelConfig = ModelConfig()


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, proj_dim)
        )
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

class AugRecHead(nn.Module):
    def __init__(self, in_dim, k_labels):
        super().__init__()
        self.fc = nn.Linear(in_dim, k_labels)
    def forward(self, x): return self.fc(x)  # logits

class DownstreamHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.fc(x)

class SSLNet(nn.Module):
    """
    Adapter that loads OxWearables backbone from torch.hub and exposes heads:
      forward(x, head={'proj','aug','cls','feats'})
    """
    def __init__(self, cfg: SSLConfig):
        super().__init__()
        self.cfg = cfg  # keep a copy for reference/saving

        # 1) load hub model, pinned to commit if provided
        h = cfg.hub
        hub_kwargs = dict(trust_repo=h.trust_repo, class_num=h.class_num, pretrained=h.pretrained, weights_only=h.weights_only)
        if h.commit is not None:
            self.hub_model: nn.Module = torch.hub.load(h.repo, h.entry, **hub_kwargs,
                                                       source="github", force_reload=h.force_reload,
                                                       skip_validation=h.skip_validation, revision=h.commit)
        else:
            self.hub_model: nn.Module = torch.hub.load(h.repo, h.entry, **hub_kwargs)

        assert hasattr(self.hub_model, "feature_extractor"), "Hub model lacks .feature_extractor"
        self.backbone: nn.Module = self.hub_model.feature_extractor

        # 2) infer feature dim from a dummy pass (no magic numbers)
        m = cfg.model
        with torch.no_grad():
            dummy = torch.zeros(1, m.in_channels, m.input_len)
            h = self.encode(dummy)
            feat_dim = int(h.shape[1])

        # 3) heads
        self.proj     = ProjectionHead(feat_dim, m.proj_dim)
        self.aug_head = AugRecHead(feat_dim, m.k_labels)
        self.cls_head = DownstreamHead(feat_dim, m.num_classes)

        if m.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)        # (B, Cb, L')
        if feats.size(-1) > 1:
            feats = feats.mean(dim=-1)  # (B, Cb)
        else:
            feats = feats.squeeze(-1)   # (B, Cb)
        return feats

    def forward(self, x: torch.Tensor, head: str = "proj"):
        h = self.encode(x)
        if head == "proj": return self.proj(h)
        if head == "aug":  return self.aug_head(h)
        if head == "cls":  return self.cls_head(h)
        if head in {"feats","h"}: return h
        raise ValueError(f"Unknown head '{head}'")

# -------------------------
# SSL losses (student focus)
# -------------------------

def augrec_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Multi-label BCE loss for augmentation recognition.
    logits: (B, K), targets: (B, K) in {0,1}
    TODO(student): consider pos_weight to handle label imbalance.
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy (SimCLR-style).
    z_i, z_j: (B, D) L2-normalized embeddings.
    Returns mean loss over positives.
    """
    B, D = z_i.size()
    z = torch.cat([z_i, z_j], dim=0)           # (2B, D)
    sim = torch.mm(z, z.t())                   # cosine sim via dot product (since normalized)

    # mask out self-similarity
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)

    # positive pairs indices: i<->i+B and i+B<->i
    pos = torch.cat([
        torch.arange(B, 2*B, device=z.device),
        torch.arange(0, B, device=z.device)
    ])
    positives = sim[torch.arange(2*B, device=z.device), pos]

    logits = sim / temperature
    labels = pos  # target index of the positive among 2B-1 negatives

    loss = F.cross_entropy(logits, labels)
    return loss

# -------------------------
# Trainer
# -------------------------

@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 4
    patience: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EarlyStopping:
    def __init__(self, patience=5):
        self.best = float("inf")
        self.count = 0
        self.patience = patience

    def step(self, val_loss):
        improved = val_loss < self.best - 1e-6
        if improved:
            self.best = val_loss
            self.count = 0
        else:
            self.count += 1
        return improved, self.count >= self.patience

# --- Metrics helper for finetune ---
def _compute_cls_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         num_classes: int,
                         label_names: List[str] | None = None) -> Dict[str, Any]:
    from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score

    # per-class
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    # aggregates
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": []
    }
    for k in range(num_classes):
        name = label_names[k] if (label_names and k < len(label_names)) else str(k)
        metrics["per_class"].append({
            "class": name,
            "precision": float(prec[k]),
            "recall": float(rec[k]),
            "f1": float(f1[k]),
            "support": int(support[k]),
        })
    return metrics

# --- Losses (as you had) ---
def augrec_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)

def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
    B, D = z_i.size()
    z = torch.cat([z_i, z_j], dim=0)           # (2B, D)
    sim = torch.mm(z, z.t())
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -9e15)
    pos = torch.cat([torch.arange(B, 2 * B, device=z.device),
                     torch.arange(0, B, device=z.device)])
    logits = sim / temperature
    labels = pos
    return F.cross_entropy(logits, labels)

# --- Trainer (records history + metrics) ---
class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.history: Dict[str, List[float]] = {}   # filled per run

    def _optim(self, model: nn.Module, lr: float):
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.cfg.weight_decay)

    def fit_augrec(self, model, train_dl: DataLoader, val_dl: DataLoader) -> Dict[str, List[float]]:
        device = self.cfg.device
        model.to(device)
        opt = self._optim(model, self.cfg.lr)
        stopper = EarlyStopping(self.cfg.patience)
        self.history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.cfg.max_epochs):
            # --- train ---
            model.train()
            train_losses = []
            pbar = tqdm(train_dl, desc=f"Epoch {epoch:03d} [train]", leave=False)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                logits = model(x, head="aug")
                loss = augrec_bce_with_logits(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                train_losses.append(loss.item())
                # live update in bar
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # --- validation ---
            model.eval()
            vloss_sum, n = 0.0, 0
            with torch.inference_mode():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    logits = model(x, head="aug")
                    vloss_sum += augrec_bce_with_logits(logits, y).item() * x.size(0)
                    n += x.size(0)
            vloss = vloss_sum / max(1, n)

            tr = float(np.mean(train_losses)) if train_losses else float("nan")
            self.history["train_loss"].append(tr)
            self.history["val_loss"].append(vloss)

            improved, stop = stopper.step(vloss)
            print(f"[AugRec] epoch {epoch:03d}  train {tr:.4f}  val {vloss:.4f}{' *' if improved else ''}")
            
            if stop:
                break

        return self.history

    def fit_contrastive(self, model, train_dl: DataLoader, val_dl: DataLoader) -> Dict[str, List[float]]:
        device = self.cfg.device
        model.to(device)
        opt = self._optim(model, self.cfg.lr)
        stopper = EarlyStopping(self.cfg.patience)
        self.history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.cfg.max_epochs):
            # --- train ---
            model.train()
            train_losses = []
            pbar = tqdm(train_dl, desc=f"Epoch {epoch:03d} [contrastive/train]", leave=False)
            for xi, xj in pbar:
                xi, xj = xi.to(device), xj.to(device)
                zi, zj = model(xi, head="proj"), model(xj, head="proj")
                loss = nt_xent_loss(zi, zj, self.cfg.temperature)
                opt.zero_grad(); loss.backward(); opt.step()
                train_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # --- validation ---
            model.eval()
            with torch.inference_mode():
                vloss_sum, n = 0.0, 0
                for xi, xj in val_dl:
                    xi, xj = xi.to(device), xj.to(device)
                    zi, zj = model(xi, head="proj"), model(xj, head="proj")
                    vloss_sum += nt_xent_loss(zi, zj, self.cfg.temperature).item() * xi.size(0)
                    n += xi.size(0)
            vloss = vloss_sum / max(1, n)

            tr = float(np.mean(train_losses)) if train_losses else float("nan")
            self.history["train_loss"].append(tr)
            self.history["val_loss"].append(vloss)

            improved, stop = stopper.step(vloss)
            print(f"[Contrastive] epoch {epoch:03d}  train {tr:.4f}  val {vloss:.4f}{' *' if improved else ''}")
            if stop:
                break

        return self.history

    def finetune(self, model, train_dl: DataLoader, val_dl: DataLoader,
                label_names: List[str] | None = None) -> Dict[str, Any]:
        device = self.cfg.device
        model.to(device)

        # freeze everything except classifier (toggle as desired)
        for p in model.backbone.parameters(): p.requires_grad = False
        for p in model.proj.parameters():     p.requires_grad = False
        for p in model.aug_head.parameters(): p.requires_grad = False
        for p in model.cls_head.parameters(): p.requires_grad = True

        opt = self._optim(model, self.cfg.lr)
        stopper = EarlyStopping(self.cfg.patience)
        ce = nn.CrossEntropyLoss()

        self.history = {"train_loss": [], "val_loss": []}
        last_metrics: Dict[str, Any] = {}

        for epoch in range(self.cfg.max_epochs):
            # --- train ---
            model.train()
            train_losses = []
            pbar = tqdm(train_dl, desc=f"Epoch {epoch:03d} [finetune/train]", leave=False)
            for x, y in pbar:
                x, y = x.to(device), y.to(device).long()
                logits = model(x, head="cls")
                loss = ce(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                train_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # --- val + metrics ---
            model.eval()
            vloss_sum, n = 0.0, 0
            all_true, all_pred = [], []
            with torch.inference_mode():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device).long()
                    logits = model(x, head="cls")
                    vloss_sum += ce(logits, y).item() * x.size(0)
                    n += x.size(0)
                    preds = logits.argmax(dim=1)
                    all_true.append(y.cpu().numpy())
                    all_pred.append(preds.cpu().numpy())
            vloss = vloss_sum / max(1, n)

            tr = float(np.mean(train_losses)) if train_losses else float("nan")
            self.history["train_loss"].append(tr)
            self.history["val_loss"].append(vloss)

            # metrics
            if all_true:
                y_true = np.concatenate(all_true)
                y_pred = np.concatenate(all_pred)
                last_metrics = _compute_cls_metrics(
                    y_true, y_pred,
                    num_classes=model.cls_head.fc.out_features,
                    label_names=label_names
                )
                print(
                    f"[Finetune] epoch {epoch:03d}  "
                    f"train {tr:.4f}  val {vloss:.4f}  "
                    f"F1(macro) {last_metrics['f1_macro']:.3f}  "
                    f"F1(weighted) {last_metrics['f1_weighted']:.3f}"
                )
            else:
                print(f"[Finetune] epoch {epoch:03d}  train {tr:.4f}  val {vloss:.4f}")

            improved, stop = stopper.step(vloss)
            if stop:
                break

        return {
            "train_loss": self.history["train_loss"],
            "val_loss": self.history["val_loss"],
            "val_metrics": last_metrics,
        }

    def evaluate(self, model, test_dl: DataLoader, label_names: List[str] | None = None) -> Dict[str, Any]:
        device = self.cfg.device
        model.eval().to(device)
        ce = nn.CrossEntropyLoss(reduction="sum")

        loss_sum, n = 0.0, 0
        all_true, all_pred = [], []
        with torch.inference_mode():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device).long()
                logits = model(x, head="cls")
                loss_sum += ce(logits, y).item()
                n += x.size(0)
                preds = logits.argmax(dim=1)
                all_true.append(y.cpu().numpy())
                all_pred.append(preds.cpu().numpy())

        avg_loss = loss_sum / max(1, n)
        y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
        y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)
        metrics = _compute_cls_metrics(
            y_true, y_pred, num_classes=model.cls_head.fc.out_features,
            label_names=label_names
        ) if y_true.size > 0 else {}

        print(f"[Test] loss {avg_loss:.4f}  "
              f"F1(macro) {metrics.get('f1_macro', float('nan')):.3f}  "
              f"F1(weighted) {metrics.get('f1_weighted', float('nan')):.3f}")

        return {"loss": avg_loss, "metrics": metrics}



# -------------------------
# Usage examples (leave commented in class; uncomment in the notebook)
# -------------------------
if __name__ == "__main__":
    pass
    # cfg = Config()
    # set_seed(cfg.seed)
    # # --- Dummy data to sanity-check pipes ---
    # N = 512
    # X = np.random.randn(N, cfg.in_channels, cfg.input_len).astype("f4")
    # y = np.random.randint(0, cfg.num_classes, size=(N,))

    # # Aug-Rec pretraining
    # aug_ds = AugRecDataset(X, augmenter=Augmenter())
    # aug_train = DataLoader(aug_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    # aug_val = DataLoader(aug_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # model = SSLNet(cfg, k_labels=len(AugRecDataset.OPS))
    # trainer = Trainer(cfg)
    # trainer.fit_augrec(model, aug_train, aug_val)

    # # Contrastive pretraining
    # con_ds = ContrastiveDataset(X, augmenter=Augmenter())
    # con_train = DataLoader(con_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    # con_val = DataLoader(con_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # model2 = SSLNet(cfg, k_labels=len(AugRecDataset.OPS))
    # trainer.fit_contrastive(model2, con_train, con_val)

    # # Finetune head on supervised labels (here: dummy)
    # sup_ds_train = BaseWearableDataset(X[:400], y[:400], augmenter=None)
    # sup_ds_val = BaseWearableDataset(X[400:], y[400:], augmenter=None)

    # # wrap to yield (x, y)
    # class SupervisedDS(BaseWearableDataset):
    #     def __getitem__(self, idx):
    #         return self._get_x(idx), torch.tensor(self.y[idx]).long()

    # train_dl = DataLoader(SupervisedDS(sup_ds_train.X, sup_ds_train.y), batch_size=cfg.batch_size, shuffle=True)
    # val_dl = DataLoader(SupervisedDS(sup_ds_val.X, sup_ds_val.y), batch_size=cfg.batch_size)

    # trainer.finetune(model2, train_dl, val_dl)
    # trainer.evaluate(model2, val_dl)

