"""
DSASR-v2 — MovieLens 1M Sensitivity Analysis
=============================================

This script is based on the supplied 1M_V3_refined.py implementation.

Purpose
-------
Run the additional experiments requested for the ablation/sensitivity
analysis on MovieLens 1M only:

    4.7.1 Temporal Window Sensitivity
    4.7.2 Fusion/Gating Mechanism Sensitivity
    4.7.3 Demographic Feature Sensitivity

No hyperparameter sensitivity study is included.

The experiments keep the main DSASR training objective and architecture
fixed wherever possible. Each configuration is trained independently
with early stopping and evaluated with the same leave-one-out + 99-negative
sampling protocol.

Outputs
-------
    ml1m_sensitivity_results.csv
    ml1m_window_sensitivity.csv
    ml1m_gating_sensitivity.csv
    ml1m_demographic_sensitivity.csv
    ml1m_sensitivity_results.tex
    ml1m_window_sensitivity.png
    ml1m_gating_sensitivity.png
    ml1m_demographic_sensitivity.png

The code does NOT perform a hyperparameter grid search.
"""

import warnings
import random
import math
import time
import os
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

# ============================================================================
# 1. GLOBAL CONFIGURATION
# ============================================================================

DATASET_NAME = "MovieLens 1M"
DATASET_TAG = "ml1m"
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main/default DSASR configuration from the supplied code
SHORT_LEN = 10
LONG_LEN = 50

MIN_HIST = 5
N_NEG_TRAIN = 5
N_NEG_EVAL = 99
BATCH = 256

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = D_MODEL * 4
DROPOUT = 0.2

LR = 1e-3
LAMBDA_BPR = 0.5
LAMBDA_L2 = 1e-4

# Baseline/full model
BASE_EPOCHS = 50
BASE_EVAL_EVERY = 5
BASE_PATIENCE = 5
BASE_VAL_USERS = 200

# Sensitivity runs
# These are intentionally shorter than the main 50-epoch run to keep the
# additional sensitivity analysis computationally manageable.
SENS_EPOCHS = 30
SENS_EVAL_EVERY = 5
SENS_PATIENCE = 4
SENS_VAL_USERS = 200

OUTPUT_DIR = Path("ml1m_sensitivity_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 78)
print("DSASR-v2 — MovieLens 1M Sensitivity Analysis")
print("=" * 78)
print(f"Device       : {DEVICE}")
print(f"PyTorch      : {torch.__version__}")
print(f"Default      : short={SHORT_LEN}, long={LONG_LEN}, d={D_MODEL}")
print(f"Output dir   : {OUTPUT_DIR.resolve()}")
print("=" * 78)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================================
# 2. DOWNLOAD / LOAD MOVIELENS 1M
# ============================================================================

DATA_ROOT = Path("ml-1m")
if not DATA_ROOT.exists():
    print("\nDownloading MovieLens 1M ...")
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(".")
    print("Download complete.")

ratings = pd.read_csv(
    DATA_ROOT / "ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

users = pd.read_csv(
    DATA_ROOT / "users.dat",
    sep="::",
    engine="python",
    names=["user_id", "gender", "age", "occupation", "zip"]
)

movies = pd.read_csv(
    DATA_ROOT / "movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movie_id", "title", "genres"]
)

print(f"Ratings : {len(ratings):,}")
print(f"Users   : {len(users):,}")
print(f"Movies  : {len(movies):,}")

# Encode demographic features exactly as in the supplied implementation.
gender_le = LabelEncoder()
occ_le = LabelEncoder()

users["gender_enc"] = gender_le.fit_transform(users["gender"])
users["occ_enc"] = occ_le.fit_transform(users["occupation"])
users["age_norm"] = MinMaxScaler().fit_transform(users[["age"]])

df = ratings.merge(
    users[["user_id", "age_norm", "gender_enc", "occ_enc"]],
    on="user_id"
)

df.sort_values(["user_id", "timestamp"], inplace=True)
df.reset_index(drop=True, inplace=True)

user_enc = LabelEncoder()
movie_enc = LabelEncoder()

df["user_idx"] = user_enc.fit_transform(df["user_id"])
df["movie_idx"] = movie_enc.fit_transform(df["movie_id"])

N_USERS = df["user_idx"].nunique()
N_MOVIES = df["movie_idx"].nunique()
N_GENDERS = df["gender_enc"].nunique()
N_OCCS = df["occ_enc"].nunique()

print(f"Encoded users  : {N_USERS:,}")
print(f"Encoded movies : {N_MOVIES:,}")
print(f"Ratings        : {len(df):,}")


# ============================================================================
# 3. USER HISTORIES AND TEMPORAL SPLIT
# ============================================================================

user_history = {}
user_meta = {}

for uid, grp in df.groupby("user_idx", sort=True):
    grp = grp.sort_values("timestamp")
    user_history[uid] = grp["movie_idx"].tolist()

    last_row = grp.iloc[-1]
    user_meta[uid] = (
        float(last_row["age_norm"]),
        int(last_row["gender_enc"]),
        int(last_row["occ_enc"])
    )

user_all_items = {uid: set(hist) for uid, hist in user_history.items()}

train_data, valid_data, test_data = [], [], []

for uid, hist in user_history.items():
    if len(hist) < MIN_HIST:
        continue

    train_hist = hist[:-2]

    for i in range(1, len(train_hist)):
        ctx = train_hist[max(0, i - LONG_LEN):i]
        train_data.append((uid, ctx, train_hist[i]))

    ctx_vt = train_hist[-LONG_LEN:]
    valid_data.append((uid, ctx_vt, hist[-2]))
    test_data.append((uid, ctx_vt, hist[-1]))

print(
    f"Train={len(train_data):,} | "
    f"Valid={len(valid_data):,} | "
    f"Test={len(test_data):,}"
)


# ============================================================================
# 4. DATASET
# ============================================================================

def pad_or_trim(seq, length):
    if len(seq) >= length:
        return seq[-length:]
    return [0] * (length - len(seq)) + list(seq)


class DSASRDataset(Dataset):
    def __init__(
        self,
        data,
        user_all_items,
        n_movies,
        short_len,
        long_len,
        n_neg,
        user_meta
    ):
        self.data = data
        self.user_items = user_all_items
        self.n_movies = n_movies
        self.short_len = short_len
        self.long_len = long_len
        self.n_neg = n_neg
        self.user_meta = user_meta

    def __len__(self):
        return len(self.data)

    def _sample_negatives(self, uid, k):
        seen = self.user_items.get(uid, set())
        negs = []
        while len(negs) < k:
            x = random.randint(0, self.n_movies - 1)
            if x not in seen:
                negs.append(x)
        return negs

    def __getitem__(self, idx):
        uid, context, pos_item = self.data[idx]

        long_seq = pad_or_trim(context, self.long_len)
        short_seq = long_seq[-self.short_len:]

        age, gender, occ = self.user_meta[uid]

        return {
            "user_idx": torch.tensor(uid, dtype=torch.long),
            "age_norm": torch.tensor([age], dtype=torch.float32),
            "gender_enc": torch.tensor(gender, dtype=torch.long),
            "occ_enc": torch.tensor(occ, dtype=torch.long),
            "long_seq": torch.tensor(long_seq, dtype=torch.long),
            "short_seq": torch.tensor(short_seq, dtype=torch.long),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "neg_items": torch.tensor(
                self._sample_negatives(uid, self.n_neg),
                dtype=torch.long
            )
        }


def make_train_loader(short_len, long_len):
    dataset = DSASRDataset(
        train_data,
        user_all_items,
        N_MOVIES,
        short_len,
        long_len,
        N_NEG_TRAIN,
        user_meta
    )
    return DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )


# ============================================================================
# 5. MODEL
# ============================================================================

class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        T = x.shape[1]

        mask = torch.triu(
            torch.ones(T, T, device=x.device),
            diagonal=1
        ).bool()

        attn_out, _ = self.attn(
            x, x, x,
            attn_mask=mask,
            need_weights=False
        )

        x = self.norm1(x + self.drop(attn_out))

        ff = self.ff2(
            self.drop(
                F.gelu(self.ff1(x))
            )
        )

        x = self.norm2(x + self.drop(ff))
        return x


class FlexibleConsistencyGate(nn.Module):
    """
    Four fusion modes are supported.

    mean:
        fixed 50/50 fusion; no learned gate.

    simple:
        learned gate using only short- and long-term representations.

    drift:
        learned gate using short + long + preference drift.

    full:
        proposed gate using short + long + drift + user context,
        including demographic features.
    """

    def __init__(self, d_model, mode="full", hidden=128):
        super().__init__()

        self.mode = mode

        if mode == "simple":
            gate_in = 2 * d_model

        elif mode == "drift":
            gate_in = 2 * d_model + 1

        elif mode == "full":
            gate_in = 2 * d_model + 1 + d_model

        elif mode == "mean":
            gate_in = None

        else:
            raise ValueError(f"Unknown gate mode: {mode}")

        if gate_in is not None:
            self.net = nn.Sequential(
                nn.Linear(gate_in, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 1),
                nn.Sigmoid()
            )

    def forward(self, short, long_repr, user_ctx=None):

        if self.mode == "mean":
            g = torch.full(
                (short.shape[0], 1),
                0.5,
                dtype=short.dtype,
                device=short.device
            )

        else:
            parts = [short, long_repr]

            if self.mode in ("drift", "full"):
                drift = (
                    1.0 -
                    F.cosine_similarity(
                        short,
                        long_repr,
                        dim=-1,
                        eps=1e-8
                    )
                ).unsqueeze(-1)

                parts.append(drift)

            if self.mode == "full":
                if user_ctx is None:
                    raise ValueError("Full gate requires user context.")
                parts.append(user_ctx)

            gate_input = torch.cat(parts, dim=-1)
            g = self.net(gate_input)

        blended = g * short + (1.0 - g) * long_repr

        return blended, g


class DSASRv2Flexible(nn.Module):

    def __init__(
        self,
        n_users,
        n_movies,
        n_genders,
        n_occs,
        short_len=10,
        long_len=50,
        gate_mode="full",
        use_demographics=True
    ):
        super().__init__()

        self.short_len = short_len
        self.long_len = long_len
        self.gate_mode = gate_mode
        self.use_demographics = use_demographics

        self.movie_emb = nn.Embedding(
            n_movies + 1,
            D_MODEL,
            padding_idx=0
        )

        self.user_emb = nn.Embedding(
            n_users + 1,
            D_MODEL
        )

        self.gender_emb = nn.Embedding(
            n_genders + 1,
            8
        )

        self.occ_emb = nn.Embedding(
            n_occs + 1,
            16
        )

        nn.init.normal_(self.movie_emb.weight, std=0.01)
        nn.init.normal_(self.user_emb.weight, std=0.01)

        self.pos_short = nn.Embedding(
            short_len + 1,
            D_MODEL
        )

        self.pos_long = nn.Embedding(
            long_len + 1,
            D_MODEL
        )

        nn.init.normal_(self.pos_short.weight, std=0.01)
        nn.init.normal_(self.pos_long.weight, std=0.01)

        self.enc_short = nn.ModuleList([
            CausalTransformerBlock(
                D_MODEL,
                N_HEADS,
                D_FF,
                DROPOUT
            )
            for _ in range(N_LAYERS)
        ])

        self.enc_long = nn.ModuleList([
            CausalTransformerBlock(
                D_MODEL,
                N_HEADS,
                D_FF,
                DROPOUT
            )
            for _ in range(N_LAYERS)
        ])

        # User context used by the proposed gate.
        # user embedding is retained in both demographic and no-demographic
        # conditions; only age/gender/occupation are removed in the ablation.
        if use_demographics:
            ctx_dim = D_MODEL + 8 + 16 + 1
        else:
            ctx_dim = D_MODEL

        self.ctx_proj = nn.Linear(ctx_dim, D_MODEL)

        self.gate = FlexibleConsistencyGate(
            D_MODEL,
            mode=gate_mode,
            hidden=D_MODEL
        )

        self.emb_dropout = nn.Dropout(DROPOUT)
        self.out_norm = nn.LayerNorm(D_MODEL)

    def encode_sequence(self, seq_ids, pos_emb, encoder_blocks):

        m_emb = self.movie_emb(seq_ids)

        pos = torch.arange(
            seq_ids.shape[1],
            device=seq_ids.device
        ).unsqueeze(0)

        p_emb = pos_emb(pos)

        x = self.emb_dropout(m_emb + p_emb)

        for block in encoder_blocks:
            x = block(x)

        return x

    def get_sequence_repr(self, batch):

        dev = next(self.parameters()).device

        long_seq = batch["long_seq"].to(dev)
        short_seq = batch["short_seq"].to(dev)

        long_enc = self.encode_sequence(
            long_seq,
            self.pos_long,
            self.enc_long
        )

        short_enc = self.encode_sequence(
            short_seq,
            self.pos_short,
            self.enc_short
        )

        long_repr = long_enc[:, -1, :]
        short_repr = short_enc[:, -1, :]

        user_ctx = None

        if self.gate_mode == "full":

            u_idx = batch["user_idx"].to(dev)

            u_emb = self.user_emb(u_idx)

            if self.use_demographics:
                g_idx = batch["gender_enc"].to(dev)
                o_idx = batch["occ_enc"].to(dev)
                age = batch["age_norm"].to(dev)

                g_emb = self.gender_emb(g_idx)
                o_emb = self.occ_emb(o_idx)

                ctx = torch.cat(
                    [u_emb, g_emb, o_emb, age],
                    dim=-1
                )
            else:
                ctx = u_emb

            user_ctx = self.ctx_proj(ctx)

        blended, gate = self.gate(
            short_repr,
            long_repr,
            user_ctx
        )

        blended = self.out_norm(blended)

        return blended, gate

    def score(self, seq_repr, item_ids):

        item_emb = self.movie_emb(item_ids)

        if item_ids.dim() == 1:
            return (seq_repr * item_emb).sum(dim=-1)

        return torch.bmm(
            item_emb,
            seq_repr.unsqueeze(-1)
        ).squeeze(-1)

    def forward(self, batch):

        blended, gate = self.get_sequence_repr(batch)

        dev = next(self.parameters()).device

        pos_score = self.score(
            blended,
            batch["pos_item"].to(dev)
        )

        neg_score = self.score(
            blended,
            batch["neg_items"].to(dev)
        )

        return pos_score, neg_score, gate


# ============================================================================
# 6. FIXED EVALUATION CANDIDATES
# ============================================================================

def build_eval_candidates(data, seed=12345):
    """
    Generate the same 99 negatives for every model/configuration.

    This is important for a fair sensitivity comparison because differences
    then come from the model rather than different negative samples.
    """
    rng = random.Random(seed)

    candidates = []

    for uid, context, pos_item in data:

        seen = user_all_items.get(uid, set())

        pool = [
            x for x in range(N_MOVIES)
            if x not in seen
        ]

        k = min(N_NEG_EVAL, len(pool))
        negs = rng.sample(pool, k)

        candidates.append(
            (uid, context, pos_item, negs)
        )

    return candidates


VALID_CANDIDATES = build_eval_candidates(valid_data, seed=12345)
TEST_CANDIDATES = build_eval_candidates(test_data, seed=54321)

print(f"Fixed validation candidates: {len(VALID_CANDIDATES):,}")
print(f"Fixed test candidates      : {len(TEST_CANDIDATES):,}")


# ============================================================================
# 7. EVALUATION
# ============================================================================

def evaluate_model(
    model,
    candidates,
    short_len,
    long_len,
    return_per_user=False
):

    model.eval()

    hr = []
    ndcg = []
    mrr = []

    with torch.no_grad():

        for uid, context, pos_item, negs in candidates:

            cands = [pos_item] + negs
            B = len(cands)

            long_seq = pad_or_trim(
                context,
                long_len
            )

            short_seq = long_seq[-short_len:]

            age, gender, occ = user_meta[uid]

            batch = {
                "user_idx": torch.tensor(
                    [uid] * B,
                    dtype=torch.long,
                    device=DEVICE
                ),

                "age_norm": torch.tensor(
                    [[age]] * B,
                    dtype=torch.float32,
                    device=DEVICE
                ),

                "gender_enc": torch.tensor(
                    [gender] * B,
                    dtype=torch.long,
                    device=DEVICE
                ),

                "occ_enc": torch.tensor(
                    [occ] * B,
                    dtype=torch.long,
                    device=DEVICE
                ),

                "long_seq": torch.tensor(
                    [long_seq] * B,
                    dtype=torch.long,
                    device=DEVICE
                ),

                "short_seq": torch.tensor(
                    [short_seq] * B,
                    dtype=torch.long,
                    device=DEVICE
                ),

                "pos_item": torch.tensor(
                    cands,
                    dtype=torch.long,
                    device=DEVICE
                ),

                "neg_items": torch.zeros(
                    B,
                    1,
                    dtype=torch.long,
                    device=DEVICE
                )
            }

            repr_, _ = model.get_sequence_repr(batch)

            scores = model.score(
                repr_,
                batch["pos_item"]
            ).detach().cpu().numpy()

            rank = 1 + int(
                (scores[1:] > scores[0]).sum()
            )

            hr.append(
                1.0 if rank <= 10 else 0.0
            )

            ndcg.append(
                1.0 / math.log2(rank + 1)
                if rank <= 10 else 0.0
            )

            mrr.append(
                1.0 / rank
                if rank <= 10 else 0.0
            )

    result = {
        "HR@10": float(np.mean(hr)),
        "NDCG@10": float(np.mean(ndcg)),
        "MRR@10": float(np.mean(mrr))
    }

    if return_per_user:
        result["per_user"] = {
            "HR@10": np.asarray(hr),
            "NDCG@10": np.asarray(ndcg),
            "MRR@10": np.asarray(mrr)
        }

    return result


# ============================================================================
# 8. TRAINING
# ============================================================================

def train_one_configuration(
    label,
    short_len,
    long_len,
    gate_mode="full",
    use_demographics=True,
    epochs=SENS_EPOCHS,
    eval_every=SENS_EVAL_EVERY,
    patience=SENS_PATIENCE,
    val_users=SENS_VAL_USERS,
    seed=SEED
):

    print("\n" + "=" * 78)
    print(f"TRAINING: {label}")
    print("=" * 78)
    print(
        f"short={short_len}, long={long_len}, "
        f"gate={gate_mode}, demographics={use_demographics}"
    )

    set_seed(seed)

    train_loader = make_train_loader(
        short_len,
        long_len
    )

    model = DSASRv2Flexible(
        n_users=N_USERS,
        n_movies=N_MOVIES,
        n_genders=N_GENDERS,
        n_occs=N_OCCS,
        short_len=short_len,
        long_len=long_len,
        gate_mode=gate_mode,
        use_demographics=use_demographics
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=0.0
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )

    best_hr = -1.0
    best_state = None
    patience_count = 0

    t0 = time.time()

    for epoch in range(1, epochs + 1):

        model.train()
        running_loss = 0.0
        batches = 0

        for batch in train_loader:

            for key in batch:
                batch[key] = batch[key].to(DEVICE)

            pos_score, neg_score, _ = model(batch)

            pos_loss = -F.logsigmoid(
                pos_score
            ).mean()

            neg_loss = -F.logsigmoid(
                -neg_score
            ).mean()

            bpr = -F.logsigmoid(
                pos_score - neg_score.mean(dim=-1)
            ).mean()

            l2 = (
                model.movie_emb.weight.norm(2) +
                model.user_emb.weight.norm(2)
            )

            loss = (
                pos_loss +
                neg_loss +
                LAMBDA_BPR * bpr +
                LAMBDA_L2 * l2
            )

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            running_loss += loss.item()
            batches += 1

        scheduler.step()

        if epoch % eval_every == 0:

            # Use a fixed subset of validation candidates for early stopping.
            val_subset = VALID_CANDIDATES[
                :min(val_users, len(VALID_CANDIDATES))
            ]

            val_result = evaluate_model(
                model,
                val_subset,
                short_len,
                long_len,
                return_per_user=False
            )

            val_hr = val_result["HR@10"]

            avg_loss = running_loss / max(batches, 1)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss={avg_loss:.4f} | "
                f"Val HR@10={val_hr:.4f}"
            )

            if val_hr > best_hr + 1e-6:

                best_hr = val_hr

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                patience_count = 0

            else:

                patience_count += 1

                if patience_count >= patience:
                    print(
                        f"Early stopping at epoch {epoch}."
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = (time.time() - t0) / 60.0

    test_result = evaluate_model(
        model,
        TEST_CANDIDATES,
        short_len,
        long_len,
        return_per_user=True
    )

    print(
        f"FINAL {label}: "
        f"HR@10={test_result['HR@10']:.4f}, "
        f"NDCG@10={test_result['NDCG@10']:.4f}, "
        f"MRR@10={test_result['MRR@10']:.4f}, "
        f"time={elapsed:.1f} min"
    )

    result = {
        "label": label,
        "short_len": short_len,
        "long_len": long_len,
        "gate_mode": gate_mode,
        "demographics": "Yes" if use_demographics else "No",
        "HR@10": test_result["HR@10"],
        "NDCG@10": test_result["NDCG@10"],
        "MRR@10": test_result["MRR@10"],
        "training_minutes": elapsed
    }

    return result, test_result["per_user"], model


# ============================================================================
# 9. 4.7.1 TEMPORAL WINDOW SENSITIVITY
# ============================================================================

print("\n")
print("#" * 78)
print("4.7.1 TEMPORAL WINDOW SENSITIVITY")
print("#" * 78)

WINDOW_CONFIGS = [
    ("W1: 5/20", 5, 20),
    ("W2: 5/50", 5, 50),
    ("W3: 10/50 (default)", 10, 50),
    ("W4: 10/100", 10, 100),
    ("W5: 20/100", 20, 100),
]

window_rows = []

for i, (label, short_len, long_len) in enumerate(
    WINDOW_CONFIGS
):

    row, _, _ = train_one_configuration(
        label=label,
        short_len=short_len,
        long_len=long_len,
        gate_mode="full",
        use_demographics=True,
        epochs=SENS_EPOCHS,
        eval_every=SENS_EVAL_EVERY,
        patience=SENS_PATIENCE,
        val_users=SENS_VAL_USERS,
        seed=SEED + 100 + i
    )

    row["study"] = "Temporal Window Sensitivity"
    window_rows.append(row)

window_df = pd.DataFrame(window_rows)

window_df.to_csv(
    OUTPUT_DIR / "ml1m_window_sensitivity.csv",
    index=False
)

print("\nTemporal window results:")
print(
    window_df[
        [
            "label",
            "HR@10",
            "NDCG@10",
            "MRR@10"
        ]
    ].to_string(index=False)
)


# ============================================================================
# 10. 4.7.2 FUSION / GATING MECHANISM SENSITIVITY
# ============================================================================

print("\n")
print("#" * 78)
print("4.7.2 FUSION / GATING MECHANISM SENSITIVITY")
print("#" * 78)

GATE_CONFIGS = [
    (
        "G1: Fixed 50/50 Mean Fusion",
        "mean",
        True
    ),
    (
        "G2: Learned Gate (short + long)",
        "simple",
        True
    ),
    (
        "G3: Drift-Aware Gate (no demographics)",
        "drift",
        False
    ),
    (
        "G4: Full Consistency Gate (proposed)",
        "full",
        True
    ),
]

gate_rows = []

for i, (label, gate_mode, use_demo) in enumerate(
    GATE_CONFIGS
):

    row, _, _ = train_one_configuration(
        label=label,
        short_len=SHORT_LEN,
        long_len=LONG_LEN,
        gate_mode=gate_mode,
        use_demographics=use_demo,
        epochs=SENS_EPOCHS,
        eval_every=SENS_EVAL_EVERY,
        patience=SENS_PATIENCE,
        val_users=SENS_VAL_USERS,
        seed=SEED + 200 + i
    )

    row["study"] = "Fusion/Gating Sensitivity"
    gate_rows.append(row)

gate_df = pd.DataFrame(gate_rows)

gate_df.to_csv(
    OUTPUT_DIR / "ml1m_gating_sensitivity.csv",
    index=False
)

print("\nFusion/gating results:")
print(
    gate_df[
        [
            "label",
            "HR@10",
            "NDCG@10",
            "MRR@10"
        ]
    ].to_string(index=False)
)


# ============================================================================
# 11. 4.7.3 DEMOGRAPHIC FEATURE SENSITIVITY
# ============================================================================

print("\n")
print("#" * 78)
print("4.7.3 DEMOGRAPHIC FEATURE SENSITIVITY")
print("#" * 78)

DEMO_CONFIGS = [
    (
        "D1: Consistency Gate without demographics",
        False
    ),
    (
        "D2: Full Consistency Gate + demographics",
        True
    ),
]

demo_rows = []

for i, (label, use_demo) in enumerate(
    DEMO_CONFIGS
):

    row, _, _ = train_one_configuration(
        label=label,
        short_len=SHORT_LEN,
        long_len=LONG_LEN,
        gate_mode="full",
        use_demographics=use_demo,
        epochs=SENS_EPOCHS,
        eval_every=SENS_EVAL_EVERY,
        patience=SENS_PATIENCE,
        val_users=SENS_VAL_USERS,
        seed=SEED + 300 + i
    )

    row["study"] = "Demographic Feature Sensitivity"
    demo_rows.append(row)

demo_df = pd.DataFrame(demo_rows)

demo_df.to_csv(
    OUTPUT_DIR / "ml1m_demographic_sensitivity.csv",
    index=False
)

print("\nDemographic feature results:")
print(
    demo_df[
        [
            "label",
            "HR@10",
            "NDCG@10",
            "MRR@10"
        ]
    ].to_string(index=False)
)


# ============================================================================
# 12. COMBINE ALL RESULTS
# ============================================================================

all_results = pd.concat(
    [
        window_df,
        gate_df,
        demo_df
    ],
    ignore_index=True
)

all_results.to_csv(
    OUTPUT_DIR / "ml1m_sensitivity_results.csv",
    index=False
)


# ============================================================================
# 13. RELATIVE CHANGE CALCULATION
# ============================================================================

def add_relative_gain(df, reference_label):

    ref = df[df["label"] == reference_label]

    if ref.empty:
        return df

    ref_hr = float(ref.iloc[0]["HR@10"])
    ref_nd = float(ref.iloc[0]["NDCG@10"])
    ref_mrr = float(ref.iloc[0]["MRR@10"])

    df = df.copy()

    df["HR_gain_%"] = (
        (df["HR@10"] - ref_hr) / ref_hr * 100.0
    )

    df["NDCG_gain_%"] = (
        (df["NDCG@10"] - ref_nd) / ref_nd * 100.0
    )

    df["MRR_gain_%"] = (
        (df["MRR@10"] - ref_mrr) / ref_mrr * 100.0
    )

    return df


window_df = add_relative_gain(
    window_df,
    "W3: 10/50 (default)"
)

gate_df = add_relative_gain(
    gate_df,
    "G4: Full Consistency Gate (proposed)"
)

demo_df = add_relative_gain(
    demo_df,
    "D2: Full Consistency Gate + demographics"
)


# ============================================================================
# 14. PUBLICATION-READY LATEX TABLES
# ============================================================================

def latex_table(
    df,
    caption,
    label,
    columns=("label", "HR@10", "NDCG@10", "MRR@10")
):

    latex = []

    latex.append(
        r"\begin{table}[!htb]"
    )
    latex.append(
        r"\centering"
    )
    latex.append(
        r"\caption{" + caption + "}"
    )
    latex.append(
        r"\label{" + label + "}"
    )
    latex.append(
        r"\begin{tabular}{lccc}"
    )
    latex.append(
        r"\toprule"
    )
    latex.append(
        r"Configuration & HR@10 & NDCG@10 & MRR@10 \\"
    )
    latex.append(
        r"\midrule"
    )

    for _, row in df.iterrows():

        name = str(row["label"]).replace(
            "_", r"\_"
        )

        latex.append(
            f"{name} & "
            f"{row['HR@10']:.4f} & "
            f"{row['NDCG@10']:.4f} & "
            f"{row['MRR@10']:.4f} \\\\"
        )

    latex.append(
        r"\bottomrule"
    )
    latex.append(
        r"\end{tabular}"
    )
    latex.append(
        r"\end{table}"
    )

    return "\n".join(latex)


latex_sections = []

latex_sections.append(
    latex_table(
        window_df,
        "Sensitivity of DSASR to short- and long-term temporal window sizes on MovieLens 1M.",
        "tab:ml1m_window_sensitivity"
    )
)

latex_sections.append("\n")

latex_sections.append(
    latex_table(
        gate_df,
        "Sensitivity of DSASR to alternative fusion and gating mechanisms on MovieLens 1M.",
        "tab:ml1m_gating_sensitivity"
    )
)

latex_sections.append("\n")

latex_sections.append(
    latex_table(
        demo_df,
        "Effect of demographic features in the Consistency Gate on MovieLens 1M.",
        "tab:ml1m_demographic_sensitivity"
    )
)

(OUTPUT_DIR / "ml1m_sensitivity_results.tex").write_text(
    "\n".join(latex_sections),
    encoding="utf-8"
)


# ============================================================================
# 15. PUBLICATION-READY PLOTS
# ============================================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "font.family": "DejaVu Sans"
})


def plot_sensitivity(
    df,
    title,
    filename,
    rotate=30
):

    metrics = ["HR@10", "NDCG@10", "MRR@10"]

    x = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    for j, metric in enumerate(metrics):

        bars = ax.bar(
            x + (j - 1) * width,
            df[metric].values,
            width,
            label=metric
        )

        for bar in bars:

            value = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.004,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            str(v).replace(":", ":\n", 1)
            if len(str(v)) > 24 else str(v)
            for v in df["label"]
        ],
        rotation=rotate,
        ha="right"
    )

    ax.set_ylabel("Ranking Score")
    ax.set_title(
        title,
        fontweight="bold"
    )

    ax.set_ylim(
        0,
        max(
            1.0,
            float(
                df[metrics].max().max()
            ) * 1.20
        )
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend(
        frameon=False,
        ncol=3
    )

    plt.tight_layout()

    path = OUTPUT_DIR / filename

    plt.savefig(
        path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


plot_sensitivity(
    window_df,
    "Temporal Window Sensitivity — MovieLens 1M",
    "ml1m_window_sensitivity.png"
)

plot_sensitivity(
    gate_df,
    "Fusion/Gating Mechanism Sensitivity — MovieLens 1M",
    "ml1m_gating_sensitivity.png"
)

plot_sensitivity(
    demo_df,
    "Demographic Feature Sensitivity — MovieLens 1M",
    "ml1m_demographic_sensitivity.png"
)


# ============================================================================
# 16. SUMMARY
# ============================================================================

print("\n")
print("=" * 78)
print("MOVIELENS 1M SENSITIVITY ANALYSIS COMPLETE")
print("=" * 78)

print("\n[4.7.1] Temporal Window Sensitivity")
print(
    window_df[
        ["label", "HR@10", "NDCG@10", "MRR@10"]
    ].to_string(index=False)
)

print("\n[4.7.2] Fusion/Gating Mechanism Sensitivity")
print(
    gate_df[
        ["label", "HR@10", "NDCG@10", "MRR@10"]
    ].to_string(index=False)
)

print("\n[4.7.3] Demographic Feature Sensitivity")
print(
    demo_df[
        ["label", "HR@10", "NDCG@10", "MRR@10"]
    ].to_string(index=False)
)

print("\nOutput files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f}")

print("\nNo hyperparameter sensitivity experiment was performed.")
print("All sensitivity experiments are restricted to MovieLens 1M.")
