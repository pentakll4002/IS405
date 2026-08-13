"""
Amazon Beauty DSASR-v2 training + statistical significance analysis.

C3: dual-stream with fixed 50/50 fusion.
C4: dual-stream with learned Consistency Gate.

The script trains both configurations, evaluates the same test users with
the same 99 negative candidates, and performs:
  * paired Wilcoxon signed-rank tests
  * Holm multiple-comparison correction
  * 10,000 paired bootstrap 95% CIs
  * relative improvements
  * paired dominance effect size
  * publication-ready CSV, LaTeX and manuscript text

Input:
    reviews_Beauty_5.json.gz

Example:
    python amazon_beauty_train_and_statistical_analysis.py \
        --data reviews_Beauty_5.json.gz
"""

import argparse
import gzip
import json
import math
import random
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


SEED = 42
SHORT_LEN = 10
LONG_LEN = 50
MIN_HIST = 5

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 512
DROPOUT = 0.20

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 3

N_NEG = 99
TOP_K = 10
N_BOOTSTRAP = 10000
ALPHA = 0.05

DEFAULT_DATA = "reviews_Beauty_5.json.gz"
OUTPUT_DIR = Path("amazon_beauty_statistical_results")

DATA_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/"
    "categoryFiles/reviews_Beauty_5.json.gz"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


# ============================================================
# Amazon Beauty data
# ============================================================

def ensure_data(path):
    path = Path(path)
    if path.exists():
        return
    print("Amazon Beauty file not found; attempting download...")
    try:
        urllib.request.urlretrieve(DATA_URL, path)
    except Exception as exc:
        raise RuntimeError(
            f"Download failed. Download reviews_Beauty_5.json.gz manually "
            f"and place it at {path}."
        ) from exc


def load_data(path):
    ensure_data(path)
    rows = []

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "reviewerID" not in d or "asin" not in d:
                continue
            rows.append({
                "user": d["reviewerID"],
                "item": d["asin"],
                "timestamp": int(d.get("unixReviewTime", 0))
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid Amazon Beauty records found.")

    df = (
        df.drop_duplicates(["user", "item"], keep="last")
          .sort_values(["user", "timestamp"])
          .reset_index(drop=True)
    )

    df["user_idx"], _ = pd.factorize(df["user"], sort=True)
    df["item_idx"], _ = pd.factorize(df["item"], sort=True)

    # Padding uses 0, therefore shift item/user indices by 1.
    df["user_idx"] += 1
    df["item_idx"] += 1

    n_users = int(df["user_idx"].nunique())
    n_items = int(df["item_idx"].nunique())

    print(f"Users: {n_users:,}")
    print(f"Items: {n_items:,}")
    print(f"Interactions: {len(df):,}")

    return df, n_users, n_items


def build_sequences(df):
    histories = {}

    for uid, g in df.groupby("user_idx", sort=True):
        h = g.sort_values("timestamp")["item_idx"].astype(int).tolist()
        if len(h) >= MIN_HIST + 2:
            histories[int(uid)] = h

    train_histories = {}
    test_data = []

    for uid, h in histories.items():
        train_histories[uid] = h[:-2]
        test_data.append((uid, h[:-1][-LONG_LEN:], h[-1]))

    all_items = {
        uid: set(h) for uid, h in histories.items()
    }

    train_examples = []
    for uid, h in train_histories.items():
        if len(h) >= MIN_HIST + 1:
            train_examples.append(
                (uid, h[:-1][-LONG_LEN:], h[-1])
            )

    print(f"Eligible/test users: {len(test_data):,}")
    print(f"Training examples: {len(train_examples):,}")

    return train_examples, test_data, all_items


def fixed_negatives(test_data, all_items, n_items):
    rng = random.Random(SEED)
    universe = set(range(1, n_items + 1))
    result = {}

    for uid, _, positive in test_data:
        pool = list(universe - all_items[uid])
        if len(pool) < N_NEG:
            raise RuntimeError(f"Not enough negatives for user {uid}.")
        result[uid] = rng.sample(pool, N_NEG)

    return result


# ============================================================
# Dataset
# ============================================================

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def pad(seq, length):
    return ([0] * max(0, length - len(seq)) + seq[-length:])


def collate(batch):
    users, long_s, short_s, pos = [], [], [], []

    for uid, context, target in batch:
        lseq = pad(context, LONG_LEN)
        sseq = lseq[-SHORT_LEN:]
        users.append(uid)
        long_s.append(lseq)
        short_s.append(sseq)
        pos.append(target)

    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(long_s, dtype=torch.long),
        torch.tensor(short_s, dtype=torch.long),
        torch.tensor(pos, dtype=torch.long),
    )


# ============================================================
# DSASR-v2
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True
        )
        self.n1 = nn.LayerNorm(D_MODEL)
        self.n2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_FF, D_MODEL)
        )
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        n = x.size(1)
        mask = torch.triu(
            torch.ones(n, n, device=x.device), diagonal=1
        ).bool()
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.n1(x + self.drop(a))
        return self.n2(x + self.drop(self.ff(x)))


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * D_MODEL + 1, D_MODEL),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D_MODEL, 1),
            nn.Sigmoid()
        )

    def forward(self, short, long):
        drift = (
            1 - F.cosine_similarity(
                short, long, dim=-1, eps=1e-8
            )
        ).unsqueeze(-1)
        g = self.net(torch.cat([short, long, drift], dim=-1))
        return g * short + (1 - g) * long, g


class DSASRv2(nn.Module):
    def __init__(self, n_users, n_items, gated=True):
        super().__init__()
        self.gated = gated

        self.item_emb = nn.Embedding(
            n_items + 1, D_MODEL, padding_idx=0
        )
        self.user_emb = nn.Embedding(
            n_users + 1, D_MODEL
        )

        self.pos_short = nn.Embedding(SHORT_LEN, D_MODEL)
        self.pos_long = nn.Embedding(LONG_LEN, D_MODEL)

        self.short_blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(N_LAYERS)]
        )
        self.long_blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(N_LAYERS)]
        )

        self.gate = Gate() if gated else None
        self.norm = nn.LayerNorm(D_MODEL)

    def encode(self, seq, pos, blocks):
        x = self.item_emb(seq)
        p = torch.arange(
            seq.size(1), device=seq.device
        )
        x = x + pos(p).unsqueeze(0)
        for block in blocks:
            x = block(x)
        return x[:, -1, :]

    def representation(self, long_seq, short_seq):
        long_r = self.encode(
            long_seq, self.pos_long, self.long_blocks
        )
        short_r = self.encode(
            short_seq, self.pos_short, self.short_blocks
        )

        if self.gated:
            fused, gate = self.gate(short_r, long_r)
        else:
            fused = 0.5 * short_r + 0.5 * long_r
            gate = torch.full(
                (fused.size(0), 1), 0.5, device=fused.device
            )

        return self.norm(fused), gate


# ============================================================
# Training
# ============================================================

def train(model, examples, epochs, batch_size, name):
    loader = torch.utils.data.DataLoader(
        TrainDataset(examples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0

        for users, long_s, short_s, pos in loader:
            long_s = long_s.to(DEVICE)
            short_s = short_s.to(DEVICE)
            pos = pos.to(DEVICE)

            optimizer.zero_grad()

            rep, _ = model.representation(long_s, short_s)

            # One random negative per training example.
            neg = torch.randint(
                1,
                model.item_emb.num_embeddings,
                pos.shape,
                device=DEVICE
            )

            pos_e = model.item_emb(pos)
            neg_e = model.item_emb(neg)

            pos_score = (rep * pos_e).sum(-1)
            neg_score = (rep * neg_e).sum(-1)

            loss = (
                F.softplus(-pos_score)
                + F.softplus(neg_score)
            ).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )
            optimizer.step()

            total += loss.item() * len(pos)
            count += len(pos)

        epoch_loss = total / max(count, 1)

        print(
            f"{name} | epoch {epoch:02d}/{epochs} | "
            f"loss={epoch_loss:.6f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"{name}: early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, test_data, negatives):
    model.eval()
    rows = []

    with torch.no_grad():
        for uid, context, positive in test_data:
            long_s = torch.tensor(
                [pad(context, LONG_LEN)],
                dtype=torch.long,
                device=DEVICE
            )
            short_s = long_s[:, -SHORT_LEN:]

            rep, gate = model.representation(
                long_s, short_s
            )

            candidates = [positive] + negatives[uid]
            cand = torch.tensor(
                candidates,
                dtype=torch.long,
                device=DEVICE
            )

            emb = model.item_emb(cand)
            scores = torch.matmul(
                emb, rep[0]
            )

            rank = 1 + int(
                (scores[1:] > scores[0]).sum().item()
            )

            hr = 1.0 if rank <= TOP_K else 0.0
            ndcg = (
                1.0 / math.log2(rank + 1)
                if rank <= TOP_K else 0.0
            )
            mrr = (
                1.0 / rank
                if rank <= TOP_K else 0.0
            )

            rows.append({
                "UserID": uid,
                "Rank": rank,
                "HR@10": hr,
                "NDCG@10": ndcg,
                "MRR@10": mrr,
                "Gate": float(gate[0, 0].item())
            })

    return pd.DataFrame(rows)


# ============================================================
# Statistics
# ============================================================

def bootstrap_ci(c4, c3, seed=SEED):
    diff = np.asarray(c4) - np.asarray(c3)
    rng = np.random.default_rng(seed)
    n = len(diff)

    boot = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        boot[i] = np.mean(
            rng.choice(diff, n, replace=True)
        )

    return (
        float(np.mean(diff)),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5))
    )


def paired_effect(c4, c3):
    diff = np.asarray(c4) - np.asarray(c3)
    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    return float(
        (np.sum(diff > 0) - np.sum(diff < 0))
        / len(diff)
    )


def statistical_analysis(c3, c4, output_dir):
    merged = c3.merge(
        c4,
        on="UserID",
        suffixes=("_C3", "_C4")
    )

    metrics = ["HR@10", "NDCG@10", "MRR@10"]
    rows = []
    p_values = []

    for metric in metrics:
        a = merged[f"{metric}_C3"].to_numpy(float)
        b = merged[f"{metric}_C4"].to_numpy(float)

        c3_mean = np.mean(a)
        c4_mean = np.mean(b)
        absolute = c4_mean - c3_mean
        relative = absolute / c3_mean * 100

        if np.allclose(a, b):
            w = 0.0
            p = 1.0
        else:
            w, p = wilcoxon(
                b, a,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto"
            )

        mean_diff, low, high = bootstrap_ci(b, a)

        rows.append({
            "Metric": metric,
            "N": len(a),
            "C3 Mean": c3_mean,
            "C4 Mean": c4_mean,
            "Absolute Gain": absolute,
            "Relative Gain (%)": relative,
            "Wilcoxon W": float(w),
            "p-value": float(p),
            "Bootstrap Mean Difference": mean_diff,
            "95% CI Lower": low,
            "95% CI Upper": high,
            "Paired Dominance Effect": paired_effect(b, a)
        })
        p_values.append(float(p))

    _, adjusted, _, _ = multipletests(
        p_values,
        alpha=ALPHA,
        method="holm"
    )

    for row, p_adj in zip(rows, adjusted):
        row["Holm-adjusted p"] = float(p_adj)
        row["Significant"] = bool(p_adj < ALPHA)

    numerical = pd.DataFrame(rows)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    numerical.to_csv(
        output_dir / "statistical_results_numerical.csv",
        index=False
    )

    publication = numerical.copy()

    publication["95% CI"] = publication.apply(
        lambda r: (
            f"[{r['95% CI Lower']:.4f}, "
            f"{r['95% CI Upper']:.4f}]"
        ),
        axis=1
    )

    for col in [
        "C3 Mean", "C4 Mean",
        "Absolute Gain"
    ]:
        publication[col] = publication[col].map(
            lambda x: f"{x:.4f}"
        )

    publication["Relative Gain (%)"] = publication[
        "Relative Gain (%)"
    ].map(lambda x: f"{x:.2f}")

    publication["p-value"] = publication["p-value"].map(
        lambda x: "<0.001" if x < 0.001 else f"{x:.4f}"
    )

    publication["Holm-adjusted p"] = publication[
        "Holm-adjusted p"
    ].map(
        lambda x: "<0.001" if x < 0.001 else f"{x:.4f}"
    )

    publication["Paired Dominance Effect"] = publication[
        "Paired Dominance Effect"
    ].map(lambda x: f"{x:.4f}")

    cols = [
        "Metric", "N", "C3 Mean", "C4 Mean",
        "Absolute Gain", "Relative Gain (%)",
        "Wilcoxon W", "p-value",
        "Holm-adjusted p", "95% CI",
        "Paired Dominance Effect", "Significant"
    ]

    publication = publication[cols]

    publication.to_csv(
        output_dir / "statistical_results_publication.csv",
        index=False
    )

    latex = publication.copy()
    latex["Significant"] = latex["Significant"].map(
        lambda x: "Yes" if x else "No"
    )

    latex_text = latex.to_latex(
        index=False,
        escape=True,
        caption=(
            "Statistical comparison of DSASR-v2 (C4) "
            "against fixed 50/50 dual-stream fusion (C3) "
            "on Amazon Beauty. Paired Wilcoxon signed-rank "
            "tests were conducted on per-user outcomes, "
            "with Holm correction for multiple comparisons. "
            "95\\% confidence intervals were obtained from "
            "10,000 paired bootstrap resamples."
        ),
        label="tab:amazon_beauty_statistical_analysis"
    )

    (output_dir / "statistical_results.tex").write_text(
        latex_text,
        encoding="utf-8"
    )

    merged.to_csv(
        output_dir / "C3_C4_paired_user_results.csv",
        index=False
    )

    paragraphs = [
        "Statistical significance was assessed using paired "
        "Wilcoxon signed-rank tests on per-user HR@10, "
        "NDCG@10, and MRR@10 outcomes. Holm correction "
        "was applied across the three metrics at α = 0.05. "
        "In addition, 10,000 paired bootstrap resamples "
        "were used to estimate 95% confidence intervals "
        "for the mean performance differences."
    ]

    for _, r in numerical.iterrows():
        p = r["p-value"]
        pa = r["Holm-adjusted p"]
        ptxt = "<0.001" if p < .001 else f"{p:.4f}"
        patxt = "<0.001" if pa < .001 else f"{pa:.4f}"

        paragraphs.append(
            f"For {r['Metric']}, C4 achieved {r['C4 Mean']:.4f} "
            f"compared with {r['C3 Mean']:.4f} for C3, "
            f"corresponding to a {r['Relative Gain (%)']:.2f}% "
            f"relative improvement (Wilcoxon p={ptxt}; "
            f"Holm-adjusted p={patxt}). The 95% bootstrap "
            f"confidence interval for the paired mean difference "
            f"was [{r['95% CI Lower']:.4f}, "
            f"{r['95% CI Upper']:.4f}]."
        )

    (output_dir / "statistical_results_publication_text.txt").write_text(
        "\n\n".join(paragraphs),
        encoding="utf-8"
    )

    print("\n" + "=" * 120)
    print("AMAZON BEAUTY STATISTICAL RESULTS")
    print("=" * 120)

    for _, r in numerical.iterrows():
        print(
            f"{r['Metric']:8s} | "
            f"C3={r['C3 Mean']:.4f} | "
            f"C4={r['C4 Mean']:.4f} | "
            f"Gain={r['Relative Gain (%)']:.2f}% | "
            f"W={r['Wilcoxon W']:.2f} | "
            f"p={r['p-value']:.4e} | "
            f"Holm p={r['Holm-adjusted p']:.4e} | "
            f"CI=[{r['95% CI Lower']:.4f}, "
            f"{r['95% CI Upper']:.4f}] | "
            f"Effect={r['Paired Dominance Effect']:.4f}"
        )

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Path to reviews_Beauty_5.json.gz"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR)
    )

    args = parser.parse_args()

    print("=" * 80)
    print("AMAZON BEAUTY — DSASR-v2 + STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    print("No previously reported aggregate results are used.")

    df, n_users, n_items = load_data(args.data)

    train_examples, test_data, all_items = build_sequences(df)

    negatives = fixed_negatives(
        test_data,
        all_items,
        n_items
    )

    # C3: fixed 50/50 fusion.
    print("\nTraining C3...")
    c3_model = DSASRv2(
        n_users, n_items, gated=False
    )
    c3_model = train(
        c3_model,
        train_examples,
        args.epochs,
        args.batch_size,
        "C3"
    )

    c3_results = evaluate(
        c3_model,
        test_data,
        negatives
    )

    # C4: learned Consistency Gate.
    print("\nTraining C4...")
    c4_model = DSASRv2(
        n_users, n_items, gated=True
    )
    c4_model = train(
        c4_model,
        train_examples,
        args.epochs,
        args.batch_size,
        "C4"
    )

    c4_results = evaluate(
        c4_model,
        test_data,
        negatives
    )

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    c3_results.to_csv(
        output_dir / "C3_per_user_results.csv",
        index=False
    )
    c4_results.to_csv(
        output_dir / "C4_per_user_results.csv",
        index=False
    )

    torch.save(
        c3_model.state_dict(),
        output_dir / "C3_model.pth"
    )
    torch.save(
        c4_model.state_dict(),
        output_dir / "C4_model.pth"
    )

    statistical_analysis(
        c3_results,
        c4_results,
        output_dir
    )

    print(
        f"\nAll results saved to: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
