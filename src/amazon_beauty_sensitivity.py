
"""
DSASR-v2 — Amazon Beauty sensitivity analysis.

Based on the supplied Amazon Beauty loader:
    reviewerID -> user
    asin -> item
    overall -> rating
    unixReviewTime -> timestamp

Experiments:
    4.7.1 Temporal Window Sensitivity
    4.7.2 Fusion/Gating Mechanism Sensitivity

No hyperparameter study is included.

The supplied Amazon Beauty data has no demographic attributes, so the
MovieLens demographic ablation is not fabricated for this dataset.
"""

import gzip, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEAUTY_PATH = "reviews_Beauty_5.json.gz"
OUT = Path("amazon_beauty_sensitivity_outputs")
OUT.mkdir(exist_ok=True)

SHORT_LEN = 10
LONG_LEN = 50

MIN_HIST = 5
N_NEG_TRAIN = 5
N_NEG_EVAL = 99
BATCH = 256

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 4 * D_MODEL
DROPOUT = 0.2

LR = 1e-3
LAMBDA_BPR = 0.5
LAMBDA_L2 = 1e-4

SENS_EPOCHS = 30
SENS_EVAL_EVERY = 5
SENS_PATIENCE = 4
SENS_VAL_USERS = 200

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()


# ============================================================
# Load Amazon Beauty
# ============================================================

print("=" * 75)
print("DSASR-v2 — Amazon Beauty Sensitivity Analysis")
print("=" * 75)
print(f"Device: {DEVICE}")
print(f"Dataset: {BEAUTY_PATH}")

records = []

with gzip.open(BEAUTY_PATH, "rt") as f:
    for line in f:
        d = json.loads(line)

        if "reviewerID" not in d:
            continue

        records.append({
            "user": d["reviewerID"],
            "item": d["asin"],
            "rating": d["overall"],
            "timestamp": d["unixReviewTime"],
        })

beauty = pd.DataFrame(records)

if beauty.empty:
    raise RuntimeError(
        f"No records loaded from {BEAUTY_PATH}"
    )

print(f"Interactions: {len(beauty):,}")


# ============================================================
# Encode users/items
# ============================================================

user_values = beauty["user"].unique()
item_values = beauty["item"].unique()

user_to_idx = {u: i for i, u in enumerate(user_values)}
item_to_idx = {x: i + 1 for i, x in enumerate(item_values)}

beauty["user_idx"] = beauty["user"].map(user_to_idx)
beauty["item_idx"] = beauty["item"].map(item_to_idx)

beauty.sort_values(
    ["user_idx", "timestamp"],
    inplace=True
)
beauty.reset_index(drop=True, inplace=True)

N_USERS = len(user_to_idx)
N_ITEMS = len(item_to_idx)

print(f"Users: {N_USERS:,}")
print(f"Items: {N_ITEMS:,}")


# ============================================================
# Build sequential train/validation/test sets
# ============================================================

user_history = {}

for uid, g in beauty.groupby("user_idx", sort=True):
    user_history[uid] = (
        g.sort_values("timestamp")["item_idx"]
        .astype(int)
        .tolist()
    )

user_all_items = {
    uid: set(hist)
    for uid, hist in user_history.items()
}

train_data = []
valid_data = []
test_data = []

for uid, hist in user_history.items():

    if len(hist) < MIN_HIST:
        continue

    train_hist = hist[:-2]

    for t in range(1, len(train_hist)):
        context = train_hist[max(0, t - LONG_LEN):t]
        train_data.append(
            (uid, context, train_hist[t])
        )

    valid_data.append(
        (uid, train_hist[-LONG_LEN:], hist[-2])
    )

    test_data.append(
        (uid, train_hist[-LONG_LEN:], hist[-1])
    )

print(
    f"Train={len(train_data):,} | "
    f"Valid={len(valid_data):,} | "
    f"Test={len(test_data):,}"
)


# ============================================================
# Utilities
# ============================================================

def pad_or_trim(seq, length):
    seq = list(seq)
    if len(seq) >= length:
        return seq[-length:]
    return [0] * (length - len(seq)) + seq


class BeautyDataset(Dataset):

    def __init__(
        self,
        data,
        histories,
        n_items,
        short_len,
        long_len,
        n_neg
    ):
        self.data = data
        self.histories = histories
        self.n_items = n_items
        self.short_len = short_len
        self.long_len = long_len
        self.n_neg = n_neg

    def __len__(self):
        return len(self.data)

    def negative_sample(self, uid):
        seen = self.histories[uid]
        result = []

        while len(result) < self.n_neg:
            x = random.randint(1, self.n_items)
            if x not in seen:
                result.append(x)

        return result

    def __getitem__(self, idx):

        uid, context, pos = self.data[idx]

        long_seq = pad_or_trim(
            context,
            self.long_len
        )

        short_seq = long_seq[-self.short_len:]

        return {
            "user_idx": torch.tensor(uid, dtype=torch.long),
            "long_seq": torch.tensor(long_seq, dtype=torch.long),
            "short_seq": torch.tensor(short_seq, dtype=torch.long),
            "pos_item": torch.tensor(pos, dtype=torch.long),
            "neg_items": torch.tensor(
                self.negative_sample(uid),
                dtype=torch.long
            ),
        }


def make_loader(short_len, long_len):

    ds = BeautyDataset(
        train_data,
        user_all_items,
        N_ITEMS,
        short_len,
        long_len,
        N_NEG_TRAIN,
    )

    return DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0
    )


# ============================================================
# DSASR Transformer
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            D_MODEL,
            N_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_FF, D_MODEL),
        )

        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):

        T = x.size(1)

        mask = torch.triu(
            torch.ones(
                T, T,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        a, _ = self.attn(
            x, x, x,
            attn_mask=mask,
            need_weights=False
        )

        x = self.norm1(
            x + self.drop(a)
        )

        x = self.norm2(
            x + self.drop(self.ff(x))
        )

        return x


class FusionGate(nn.Module):

    def __init__(self, mode):

        super().__init__()
        self.mode = mode

        if mode == "mean":
            self.net = None

        elif mode == "simple":
            self.net = nn.Sequential(
                nn.Linear(2 * D_MODEL, D_MODEL),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(D_MODEL, 1),
                nn.Sigmoid(),
            )

        elif mode in ("drift", "full"):
            self.net = nn.Sequential(
                nn.Linear(2 * D_MODEL + 1, D_MODEL),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(D_MODEL, 1),
                nn.Sigmoid(),
            )

        else:
            raise ValueError(mode)

    def forward(self, short, long):

        if self.mode == "mean":

            gate = torch.full(
                (short.size(0), 1),
                0.5,
                device=short.device,
                dtype=short.dtype,
            )

        elif self.mode == "simple":

            gate = self.net(
                torch.cat([short, long], dim=-1)
            )

        else:

            drift = (
                1.0
                - F.cosine_similarity(
                    short,
                    long,
                    dim=-1,
                    eps=1e-8,
                )
            ).unsqueeze(-1)

            gate = self.net(
                torch.cat(
                    [short, long, drift],
                    dim=-1
                )
            )

        fused = (
            gate * short
            + (1.0 - gate) * long
        )

        return fused, gate


class DSASRBeauty(nn.Module):

    def __init__(
        self,
        n_users,
        n_items,
        short_len,
        long_len,
        gate_mode,
    ):

        super().__init__()

        self.movie_emb = nn.Embedding(
            n_items + 1,
            D_MODEL,
            padding_idx=0,
        )

        self.user_emb = nn.Embedding(
            n_users,
            D_MODEL,
        )

        self.pos_short = nn.Embedding(
            short_len,
            D_MODEL
        )

        self.pos_long = nn.Embedding(
            long_len,
            D_MODEL
        )

        self.short_encoder = nn.ModuleList(
            [TransformerBlock() for _ in range(N_LAYERS)]
        )

        self.long_encoder = nn.ModuleList(
            [TransformerBlock() for _ in range(N_LAYERS)]
        )

        self.gate = FusionGate(gate_mode)

        self.norm = nn.LayerNorm(D_MODEL)

    def encode(
        self,
        seq,
        pos,
        blocks
    ):

        positions = torch.arange(
            seq.size(1),
            device=seq.device
        ).unsqueeze(0)

        x = (
            self.movie_emb(seq)
            + pos(positions)
        )

        for block in blocks:
            x = block(x)

        return x

    def get_sequence_repr(self, batch):

        long_enc = self.encode(
            batch["long_seq"],
            self.pos_long,
            self.long_encoder
        )

        short_enc = self.encode(
            batch["short_seq"],
            self.pos_short,
            self.short_encoder
        )

        long_repr = long_enc[:, -1, :]
        short_repr = short_enc[:, -1, :]

        fused, gate = self.gate(
            short_repr,
            long_repr
        )

        return self.norm(fused), gate

    def score(self, representation, item_ids):

        item_emb = self.movie_emb(item_ids)

        if item_ids.dim() == 1:
            return (
                representation * item_emb
            ).sum(-1)

        return torch.bmm(
            item_emb,
            representation.unsqueeze(-1)
        ).squeeze(-1)

    def forward(self, batch):

        rep, gate = self.get_sequence_repr(batch)

        pos_score = self.score(
            rep,
            batch["pos_item"]
        )

        neg_score = self.score(
            rep,
            batch["neg_items"]
        )

        return pos_score, neg_score, gate


# ============================================================
# Fixed 99-negative evaluation protocol
# ============================================================

def build_candidates(data, seed):

    rng = random.Random(seed)

    candidates = []

    all_items = set(
        range(1, N_ITEMS + 1)
    )

    for uid, context, pos in data:

        pool = list(
            all_items - user_all_items[uid]
        )

        k = min(
            N_NEG_EVAL,
            len(pool)
        )

        negatives = rng.sample(
            pool,
            k
        )

        candidates.append(
            (uid, context, pos, negatives)
        )

    return candidates


VALID_CANDIDATES = build_candidates(
    valid_data,
    12345
)

TEST_CANDIDATES = build_candidates(
    test_data,
    54321
)


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    candidates,
    short_len,
    long_len
):

    model.eval()

    hr = []
    ndcg = []
    mrr = []

    with torch.no_grad():

        for uid, context, pos, negatives in candidates:

            items = [pos] + negatives
            n = len(items)

            long_seq = pad_or_trim(
                context,
                long_len
            )

            short_seq = long_seq[-short_len:]

            batch = {
                "user_idx": torch.full(
                    (n,),
                    uid,
                    dtype=torch.long,
                    device=DEVICE,
                ),

                "long_seq": torch.tensor(
                    [long_seq] * n,
                    dtype=torch.long,
                    device=DEVICE,
                ),

                "short_seq": torch.tensor(
                    [short_seq] * n,
                    dtype=torch.long,
                    device=DEVICE,
                ),

                "pos_item": torch.tensor(
                    items,
                    dtype=torch.long,
                    device=DEVICE,
                ),

                "neg_items": torch.zeros(
                    n, 1,
                    dtype=torch.long,
                    device=DEVICE,
                ),
            }

            rep, _ = model.get_sequence_repr(
                batch
            )

            scores = model.score(
                rep,
                batch["pos_item"]
            ).cpu().numpy()

            rank = 1 + int(
                (scores[1:] > scores[0]).sum()
            )

            hr.append(
                float(rank <= 10)
            )

            ndcg.append(
                1.0 / math.log2(rank + 1)
                if rank <= 10 else 0.0
            )

            mrr.append(
                1.0 / rank
                if rank <= 10 else 0.0
            )

    return {
        "HR@10": float(np.mean(hr)),
        "NDCG@10": float(np.mean(ndcg)),
        "MRR@10": float(np.mean(mrr)),
    }


# ============================================================
# Training
# ============================================================

def train_config(
    label,
    short_len,
    long_len,
    gate_mode,
    seed
):

    print("\n" + "=" * 75)
    print(label)
    print(
        f"short={short_len}, "
        f"long={long_len}, "
        f"gate={gate_mode}"
    )
    print("=" * 75)

    seed_everything(seed)

    loader = make_loader(
        short_len,
        long_len
    )

    model = DSASRBeauty(
        N_USERS,
        N_ITEMS,
        short_len,
        long_len,
        gate_mode,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SENS_EPOCHS,
        eta_min=1e-5
    )

    best_hr = -1
    best_state = None
    patience_count = 0

    start = time.time()

    for epoch in range(1, SENS_EPOCHS + 1):

        model.train()
        total_loss = 0
        n_batches = 0

        for batch in loader:

            batch = {
                k: v.to(DEVICE)
                for k, v in batch.items()
            }

            pos, neg, _ = model(batch)

            pos_loss = -F.logsigmoid(pos).mean()
            neg_loss = -F.logsigmoid(-neg).mean()

            bpr = -F.logsigmoid(
                pos - neg.mean(dim=-1)
            ).mean()

            l2 = (
                model.movie_emb.weight.norm(2)
                + model.user_emb.weight.norm(2)
            )

            loss = (
                pos_loss
                + neg_loss
                + LAMBDA_BPR * bpr
                + LAMBDA_L2 * l2
            )

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % SENS_EVAL_EVERY == 0:

            subset = VALID_CANDIDATES[
                :min(
                    SENS_VAL_USERS,
                    len(VALID_CANDIDATES)
                )
            ]

            val = evaluate(
                model,
                subset,
                short_len,
                long_len
            )

            avg_loss = (
                total_loss / max(n_batches, 1)
            )

            print(
                f"Epoch {epoch:02d}/{SENS_EPOCHS} | "
                f"Loss={avg_loss:.4f} | "
                f"Val HR@10={val['HR@10']:.4f}"
            )

            if val["HR@10"] > best_hr + 1e-6:

                best_hr = val["HR@10"]

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                patience_count = 0

            else:

                patience_count += 1

                if patience_count >= SENS_PATIENCE:
                    print("Early stopping.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(
        model,
        TEST_CANDIDATES,
        short_len,
        long_len
    )

    minutes = (
        time.time() - start
    ) / 60

    print(
        f"FINAL | HR@10={test['HR@10']:.4f} | "
        f"NDCG@10={test['NDCG@10']:.4f} | "
        f"MRR@10={test['MRR@10']:.4f}"
    )

    return {
        "label": label,
        "short_len": short_len,
        "long_len": long_len,
        "gate_mode": gate_mode,
        "HR@10": test["HR@10"],
        "NDCG@10": test["NDCG@10"],
        "MRR@10": test["MRR@10"],
        "training_minutes": minutes,
    }


# ============================================================
# 4.7.1 Temporal Window Sensitivity
# ============================================================

WINDOWS = [
    ("W1: 5/20", 5, 20),
    ("W2: 5/50", 5, 50),
    ("W3: 10/50 (default)", 10, 50),
    ("W4: 10/100", 10, 100),
    ("W5: 20/100", 20, 100),
]

window_results = []

for i, (label, short_len, long_len) in enumerate(WINDOWS):

    row = train_config(
        label,
        short_len,
        long_len,
        "full",
        SEED + 100 + i
    )

    row["study"] = "Temporal Window Sensitivity"
    window_results.append(row)

window_df = pd.DataFrame(window_results)

window_df.to_csv(
    OUT / "amazon_beauty_window_sensitivity.csv",
    index=False
)


# ============================================================
# 4.7.2 Fusion/Gating Sensitivity
# ============================================================

GATES = [
    ("G1: Fixed 50/50 Mean Fusion", "mean"),
    ("G2: Learned Gate (short + long)", "simple"),
    ("G3: Drift-Aware Gate", "drift"),
    ("G4: Full Consistency Gate (proposed)", "full"),
]

gate_results = []

for i, (label, mode) in enumerate(GATES):

    row = train_config(
        label,
        SHORT_LEN,
        LONG_LEN,
        mode,
        SEED + 200 + i
    )

    row["study"] = "Fusion/Gating Sensitivity"
    gate_results.append(row)

gate_df = pd.DataFrame(gate_results)

gate_df.to_csv(
    OUT / "amazon_beauty_gating_sensitivity.csv",
    index=False
)


# ============================================================
# Combined results
# ============================================================

results = pd.concat(
    [window_df, gate_df],
    ignore_index=True
)

results.to_csv(
    OUT / "amazon_beauty_sensitivity_results.csv",
    index=False
)


# ============================================================
# Relative improvement over the proposed/default model
# ============================================================

def relative_gain(df, reference):

    df = df.copy()

    ref = df[
        df["label"] == reference
    ]

    if ref.empty:
        return df

    for metric in [
        "HR@10",
        "NDCG@10",
        "MRR@10"
    ]:

        base = float(
            ref.iloc[0][metric]
        )

        df[metric + "_gain_%"] = (
            (df[metric] - base)
            / base * 100
            if base != 0
            else np.nan
        )

    return df


window_df = relative_gain(
    window_df,
    "W3: 10/50 (default)"
)

gate_df = relative_gain(
    gate_df,
    "G4: Full Consistency Gate (proposed)"
)


# ============================================================
# LaTeX tables
# ============================================================

def latex_table(df, caption, label):

    lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Configuration & HR@10 & NDCG@10 & MRR@10 \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():

        name = (
            str(row["label"])
            .replace("_", r"\_")
            .replace("&", r"\&")
        )

        lines.append(
            f"{name} & "
            f"{row['HR@10']:.4f} & "
            f"{row['NDCG@10']:.4f} & "
            f"{row['MRR@10']:.4f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


latex = "\n\n".join([
    latex_table(
        window_df,
        "Temporal window sensitivity of DSASR on Amazon Beauty.",
        "tab:beauty_window_sensitivity",
    ),
    latex_table(
        gate_df,
        "Fusion and gating mechanism sensitivity of DSASR on Amazon Beauty.",
        "tab:beauty_gating_sensitivity",
    ),
])

(
    OUT / "amazon_beauty_sensitivity_results.tex"
).write_text(
    latex,
    encoding="utf-8"
)


# ============================================================
# Publication-ready plots
# ============================================================

def plot_results(df, title, filename):

    metrics = [
        "HR@10",
        "NDCG@10",
        "MRR@10"
    ]

    x = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    for j, metric in enumerate(metrics):

        bars = ax.bar(
            x + (j - 1) * width,
            df[metric],
            width,
            label=metric,
        )

        for bar in bars:

            h = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                h + 0.004,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            str(x).replace(
                ":",
                ":\n",
                1
            )
            if len(str(x)) > 24
            else str(x)
            for x in df["label"]
        ],
        rotation=25,
        ha="right",
    )

    ax.set_ylabel("Ranking Score")
    ax.set_title(title, fontweight="bold")

    ymax = float(
        df[metrics].max().max()
    )

    ax.set_ylim(
        0,
        max(1.0, ymax * 1.2)
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

    plt.savefig(
        OUT / filename,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()


plot_results(
    window_df,
    "Temporal Window Sensitivity — Amazon Beauty",
    "amazon_beauty_window_sensitivity.png",
)

plot_results(
    gate_df,
    "Fusion/Gating Sensitivity — Amazon Beauty",
    "amazon_beauty_gating_sensitivity.png",
)


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 75)
print("AMAZON BEAUTY SENSITIVITY ANALYSIS COMPLETE")
print("=" * 75)

print("\nTemporal Window Sensitivity")
print(
    window_df[
        ["label", "HR@10", "NDCG@10", "MRR@10"]
    ].to_string(index=False)
)

print("\nFusion/Gating Sensitivity")
print(
    gate_df[
        ["label", "HR@10", "NDCG@10", "MRR@10"]
    ].to_string(index=False)
)

print("\nOutput files:")
for f in sorted(OUT.iterdir()):
    print(" ", f)

print(
    "\nNo hyperparameter sensitivity study was performed."
)
print(
    "Demographic-feature sensitivity was omitted because "
    "the supplied Amazon Beauty schema contains no demographic attributes."
)
