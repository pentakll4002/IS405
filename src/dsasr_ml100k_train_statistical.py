
"""
DSASR-v2 — MovieLens 100K
Training + Publication-Ready Statistical Analysis
=================================================

This standalone script is adapted from the uploaded
DSASR_ImplementationMovieLens100.ipynb.

C3:
    Dual-stream short/long encoders + fixed 50/50 fusion
    + BCE + BPR

C4:
    Dual-stream short/long encoders + learned Consistency Gate
    + user context + BCE + BPR

Statistical comparison:
    * Per-user HR@10
    * Per-user NDCG@10
    * Per-user MRR@10
    * Paired Wilcoxon signed-rank test
    * Holm correction for 3 metrics
    * 10,000 paired bootstrap resamples
    * 95% confidence intervals
    * Paired dominance effect size
    * Publication-ready CSV
    * Publication-ready LaTeX
    * Manuscript-ready statistical paragraph

IMPORTANT
---------
This is a NEW training/evaluation run. It does not infer p-values from
aggregate means. C3 and C4 are trained independently and then evaluated
on exactly the SAME test users and exactly the SAME 99 negative candidates.

Dataset:
    MovieLens 100K

Input:
    ml-100k/u.data
    ml-100k/u.user
    ml-100k/u.item

If ml-100k is absent, the script attempts to download it.

Example:
    python dsasr_ml100k_train_statistical.py

For a faster smoke test:
    python dsasr_ml100k_train_statistical.py --epochs 5

Recommended paper run:
    python dsasr_ml100k_train_statistical.py --epochs 50
"""

import argparse
import io
import math
import random
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ============================================================
# Configuration
# ============================================================

SEED = 42

DATASET_NAME = "MovieLens 100K"
DATASET_TAG = "ml100k"

SHORT_LEN = 10
LONG_LEN = 50
MIN_HIST = 5

N_NEG_TRAIN = 5
N_NEG_EVAL = 99

BATCH_SIZE = 256

D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
D_FF = D_MODEL * 4
DROPOUT = 0.2

LR = 1e-3
LAMBDA_BPR = 0.5
LAMBDA_L2 = 1e-4

DEFAULT_EPOCHS = 50
EVAL_EVERY = 5
PATIENCE = 5
VAL_USERS = 200

TOP_K = 10

N_BOOTSTRAP = 10000
ALPHA = 0.05

OUTPUT_DIR = Path(
    "ml100k_statistical_results"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

warnings.filterwarnings("ignore")


# ============================================================
# Reproducibility
# ============================================================

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
# Download/load MovieLens 100K
# ============================================================

def ensure_dataset():

    data_dir = Path("ml-100k")

    required = [
        data_dir / "u.data",
        data_dir / "u.user",
        data_dir / "u.item",
    ]

    if all(path.exists() for path in required):
        print("MovieLens 100K dataset found.")
        return

    print("Downloading MovieLens 100K...")

    url = (
        "https://files.grouplens.org/"
        "datasets/movielens/ml-100k.zip"
    )

    response = requests.get(
        url,
        timeout=120
    )
    response.raise_for_status()

    with zipfile.ZipFile(
        io.BytesIO(response.content)
    ) as z:
        z.extractall(".")

    print("MovieLens 100K download complete.")


def load_dataset():

    ensure_dataset()

    ratings = pd.read_csv(
        "ml-100k/u.data",
        sep="\t",
        names=[
            "user_id",
            "movie_id",
            "rating",
            "timestamp",
        ],
    )

    users = pd.read_csv(
        "ml-100k/u.user",
        sep="|",
        names=[
            "user_id",
            "age",
            "gender",
            "occupation",
            "zip_code",
        ],
    )

    genre_cols = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    movies = pd.read_csv(
        "ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        names=[
            "movie_id",
            "title",
            "release_date",
            "video_date",
            "imdb_url",
        ] + genre_cols,
    )

    # Same demographic preprocessing as the uploaded notebook.
    gender_le = LabelEncoder()
    occupation_le = LabelEncoder()

    users["gender_enc"] = (
        gender_le.fit_transform(
            users["gender"]
        )
    )

    users["occ_enc"] = (
        occupation_le.fit_transform(
            users["occupation"]
        )
    )

    users["age_norm"] = (
        MinMaxScaler().fit_transform(
            users[["age"]]
        )
    )

    df = ratings.merge(
        users[
            [
                "user_id",
                "age_norm",
                "gender_enc",
                "occ_enc",
            ]
        ],
        on="user_id",
    )

    df = df.merge(
        movies[
            ["movie_id"] + genre_cols
        ],
        on="movie_id",
        how="left",
    )

    df.sort_values(
        ["user_id", "timestamp"],
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    df["user_idx"] = (
        user_encoder.fit_transform(
            df["user_id"]
        )
    )

    df["movie_idx"] = (
        movie_encoder.fit_transform(
            df["movie_id"]
        )
    )

    n_users = int(
        df["user_idx"].nunique()
    )

    n_movies = int(
        df["movie_idx"].nunique()
    )

    n_genders = int(
        df["gender_enc"].nunique()
    )

    n_occs = int(
        df["occ_enc"].nunique()
    )

    print()
    print(
        f"Users  : {n_users:,}"
    )
    print(
        f"Movies : {n_movies:,}"
    )
    print(
        f"Ratings: {len(df):,}"
    )
    print(
        f"Avg interactions/user: "
        f"{len(df)/n_users:.1f}"
    )
    print(
        f"Sparsity: "
        f"{1 - len(df)/(n_users*n_movies):.4%}"
    )

    return (
        df,
        n_users,
        n_movies,
        n_genders,
        n_occs,
    )


# ============================================================
# Sequence construction
# ============================================================

def build_sequences(df):

    user_history = {}
    user_meta = {}

    for uid, group in df.groupby(
        "user_idx",
        sort=True,
    ):

        history = (
            group.sort_values(
                "timestamp"
            )["movie_idx"]
            .astype(int)
            .tolist()
        )

        user_history[
            int(uid)
        ] = history

        last_row = group.iloc[-1]

        user_meta[
            int(uid)
        ] = (
            float(last_row["age_norm"]),
            int(last_row["gender_enc"]),
            int(last_row["occ_enc"]),
        )

    user_all_items = {
        uid: set(history)
        for uid, history
        in user_history.items()
    }

    train_data = []
    valid_data = []
    test_data = []

    for uid, history in user_history.items():

        if len(history) < MIN_HIST:
            continue

        train_hist = history[:-2]

        for i in range(
            1,
            len(train_hist)
        ):

            context = train_hist[
                max(
                    0,
                    i - LONG_LEN
                ):i
            ]

            train_data.append(
                (
                    uid,
                    context,
                    train_hist[i],
                )
            )

        valid_context = (
            train_hist[-LONG_LEN:]
        )

        valid_data.append(
            (
                uid,
                valid_context,
                history[-2],
            )
        )

        test_data.append(
            (
                uid,
                valid_context,
                history[-1],
            )
        )

    print()
    print(
        f"Train : {len(train_data):,}"
    )
    print(
        f"Valid : {len(valid_data):,} "
        "(one per user)"
    )
    print(
        f"Test  : {len(test_data):,} "
        "(one per user)"
    )

    return (
        user_history,
        user_meta,
        user_all_items,
        train_data,
        valid_data,
        test_data,
    )


# ============================================================
# Padding
# ============================================================

def pad_or_trim(
    seq,
    length,
):

    if len(seq) >= length:
        return seq[-length:]

    return (
        [0] * (
            length - len(seq)
        )
        + list(seq)
    )


# ============================================================
# Training dataset
# ============================================================

class DSASRv2Dataset(
    Dataset
):

    def __init__(
        self,
        data,
        user_all_items,
        n_movies,
        short_len,
        long_len,
        n_neg,
        user_meta,
    ):

        self.data = data
        self.user_items = (
            user_all_items
        )
        self.n_movies = n_movies
        self.short_len = short_len
        self.long_len = long_len
        self.n_neg = n_neg
        self.user_meta = user_meta

    def __len__(self):
        return len(self.data)

    def sample_negatives(
        self,
        uid,
        k,
    ):

        seen = self.user_items.get(
            uid,
            set(),
        )

        negatives = []

        while len(
            negatives
        ) < k:

            neg = random.randint(
                0,
                self.n_movies - 1,
            )

            if neg not in seen:
                negatives.append(
                    neg
                )

        return negatives

    def __getitem__(
        self,
        index,
    ):

        uid, context, pos_item = (
            self.data[index]
        )

        negatives = (
            self.sample_negatives(
                uid,
                self.n_neg,
            )
        )

        long_seq = pad_or_trim(
            context,
            self.long_len,
        )

        short_seq = long_seq[
            -self.short_len:
        ]

        age, gender, occ = (
            self.user_meta[uid]
        )

        return {
            "user_idx": torch.tensor(
                uid,
                dtype=torch.long,
            ),
            "age_norm": torch.tensor(
                [age],
                dtype=torch.float32,
            ),
            "gender_enc": torch.tensor(
                gender,
                dtype=torch.long,
            ),
            "occ_enc": torch.tensor(
                occ,
                dtype=torch.long,
            ),
            "long_seq": torch.tensor(
                long_seq,
                dtype=torch.long,
            ),
            "short_seq": torch.tensor(
                short_seq,
                dtype=torch.long,
            ),
            "pos_item": torch.tensor(
                pos_item,
                dtype=torch.long,
            ),
            "neg_items": torch.tensor(
                negatives,
                dtype=torch.long,
            ),
        }


# ============================================================
# Transformer block
# ============================================================

class CausalTransformerBlock(
    nn.Module
):

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout,
    ):

        super().__init__()

        self.attn = (
            nn.MultiheadAttention(
                d_model,
                n_heads,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.ff1 = nn.Linear(
            d_model,
            d_ff,
        )

        self.ff2 = nn.Linear(
            d_ff,
            d_model,
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.drop = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        causal=True,
    ):

        _, T, _ = x.shape

        if causal:

            mask = torch.triu(
                torch.ones(
                    T,
                    T,
                    device=x.device,
                ),
                diagonal=1,
            ).bool()

        else:
            mask = None

        attn_out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=mask,
        )

        x = self.norm1(
            x
            + self.drop(
                attn_out
            )
        )

        ff = self.ff2(
            self.drop(
                F.gelu(
                    self.ff1(x)
                )
            )
        )

        x = self.norm2(
            x
            + self.drop(ff)
        )

        return x


# ============================================================
# Consistency Gate
# ============================================================

class ConsistencyGate(
    nn.Module
):

    def __init__(
        self,
        d_model,
        ctx_dim,
        hidden=128,
    ):

        super().__init__()

        gate_in = (
            d_model * 2
            + 1
            + ctx_dim
        )

        self.net = nn.Sequential(
            nn.Linear(
                gate_in,
                hidden,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                hidden,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        short,
        long_,
        user_ctx,
    ):

        drift = (
            1.0
            - F.cosine_similarity(
                short,
                long_,
                dim=-1,
                eps=1e-8,
            )
        ).unsqueeze(-1)

        gate_input = torch.cat(
            [
                short,
                long_,
                drift,
                user_ctx,
            ],
            dim=-1,
        )

        gate = self.net(
            gate_input
        )

        blended = (
            gate * short
            + (1.0 - gate)
            * long_
        )

        return (
            blended,
            gate,
        )


# ============================================================
# DSASR-v2
# ============================================================

class DSASRv2(
    nn.Module
):

    def __init__(
        self,
        n_users,
        n_movies,
        n_genders,
        n_occs,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        short_len=SHORT_LEN,
        long_len=LONG_LEN,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.d_model = d_model
        self.short_len = short_len
        self.long_len = long_len

        self.movie_emb = nn.Embedding(
            n_movies + 1,
            d_model,
            padding_idx=0,
        )

        self.user_emb = nn.Embedding(
            n_users + 1,
            d_model,
        )

        self.gender_emb = nn.Embedding(
            n_genders + 1,
            8,
        )

        self.occ_emb = nn.Embedding(
            n_occs + 1,
            16,
        )

        nn.init.normal_(
            self.movie_emb.weight,
            std=0.01,
        )

        nn.init.normal_(
            self.user_emb.weight,
            std=0.01,
        )

        self.pos_short = nn.Embedding(
            short_len + 1,
            d_model,
        )

        self.pos_long = nn.Embedding(
            long_len + 1,
            d_model,
        )

        nn.init.normal_(
            self.pos_short.weight,
            std=0.01,
        )

        nn.init.normal_(
            self.pos_long.weight,
            std=0.01,
        )

        self.enc_short = (
            nn.ModuleList(
                [
                    CausalTransformerBlock(
                        d_model,
                        n_heads,
                        d_ff,
                        dropout,
                    )
                    for _ in range(
                        n_layers
                    )
                ]
            )
        )

        self.enc_long = (
            nn.ModuleList(
                [
                    CausalTransformerBlock(
                        d_model,
                        n_heads,
                        d_ff,
                        dropout,
                    )
                    for _ in range(
                        n_layers
                    )
                ]
            )
        )

        ctx_dim = (
            d_model
            + 8
            + 16
            + 1
        )

        self.ctx_proj = nn.Linear(
            ctx_dim,
            d_model,
        )

        self.gate = ConsistencyGate(
            d_model,
            d_model,
            hidden=d_model,
        )

        self.emb_dropout = nn.Dropout(
            dropout
        )

        self.out_norm = nn.LayerNorm(
            d_model
        )

    def encode_sequence(
        self,
        seq_ids,
        pos_emb,
        encoder_blocks,
        causal=True,
    ):

        m_emb = self.movie_emb(
            seq_ids
        )

        T = seq_ids.shape[1]

        pos = torch.arange(
            T,
            device=seq_ids.device,
        ).unsqueeze(0)

        p_emb = pos_emb(
            pos
        )

        x = self.emb_dropout(
            m_emb + p_emb
        )

        for block in encoder_blocks:
            x = block(
                x,
                causal=causal,
            )

        return x

    def get_sequence_repr(
        self,
        batch,
    ):

        dev = next(
            self.parameters()
        ).device

        long_seq = batch[
            "long_seq"
        ].to(dev)

        short_seq = batch[
            "short_seq"
        ].to(dev)

        u_idx = batch[
            "user_idx"
        ].to(dev)

        g_idx = batch[
            "gender_enc"
        ].to(dev)

        o_idx = batch[
            "occ_enc"
        ].to(dev)

        age = batch[
            "age_norm"
        ].to(dev)

        long_enc = self.encode_sequence(
            long_seq,
            self.pos_long,
            self.enc_long,
            causal=True,
        )

        short_enc = self.encode_sequence(
            short_seq,
            self.pos_short,
            self.enc_short,
            causal=True,
        )

        long_repr = long_enc[
            :, -1, :
        ]

        short_repr = short_enc[
            :, -1, :
        ]

        u_emb = self.user_emb(
            u_idx
        )

        g_emb = self.gender_emb(
            g_idx
        )

        o_emb = self.occ_emb(
            o_idx
        )

        user_ctx = torch.cat(
            [
                u_emb,
                g_emb,
                o_emb,
                age,
            ],
            dim=-1,
        )

        user_ctx_proj = (
            self.ctx_proj(
                user_ctx
            )
        )

        blended, gate = (
            self.gate(
                short_repr,
                long_repr,
                user_ctx_proj,
            )
        )

        blended = self.out_norm(
            blended
        )

        return (
            blended,
            gate,
        )

    def score(
        self,
        seq_repr,
        item_ids,
    ):

        if item_ids.dim() == 1:

            item_emb = self.movie_emb(
                item_ids
            )

            return (
                seq_repr
                * item_emb
            ).sum(dim=-1)

        item_emb = self.movie_emb(
            item_ids
        )

        return torch.bmm(
            item_emb,
            seq_repr.unsqueeze(-1),
        ).squeeze(-1)

    def forward(
        self,
        batch,
    ):

        blended, gate = (
            self.get_sequence_repr(
                batch
            )
        )

        dev = next(
            self.parameters()
        ).device

        pos_score = self.score(
            blended,
            batch[
                "pos_item"
            ].to(dev),
        )

        neg_score = self.score(
            blended,
            batch[
                "neg_items"
            ].to(dev),
        )

        return (
            pos_score,
            neg_score,
            gate,
        )


# ============================================================
# C3: fixed 50/50 dual-stream fusion
# ============================================================

class DSASRv2_C3(
    DSASRv2
):

    def get_sequence_repr(
        self,
        batch,
    ):

        dev = next(
            self.parameters()
        ).device

        long_seq = batch[
            "long_seq"
        ].to(dev)

        short_seq = batch[
            "short_seq"
        ].to(dev)

        long_enc = self.encode_sequence(
            long_seq,
            self.pos_long,
            self.enc_long,
            causal=True,
        )

        short_enc = self.encode_sequence(
            short_seq,
            self.pos_short,
            self.enc_short,
            causal=True,
        )

        long_repr = long_enc[
            :, -1, :
        ]

        short_repr = short_enc[
            :, -1, :
        ]

        blended = self.out_norm(
            0.5 * short_repr
            + 0.5 * long_repr
        )

        gate = torch.zeros(
            blended.shape[0],
            1,
            device=dev,
        )

        return (
            blended,
            gate,
        )


# ============================================================
# Loss
# ============================================================

def compute_loss(
    model,
    pos_score,
    neg_score,
):

    pos_loss = (
        -F.logsigmoid(
            pos_score
        ).mean()
    )

    neg_loss = (
        -F.logsigmoid(
            -neg_score
        ).mean()
    )

    bpr = (
        -F.logsigmoid(
            pos_score
            - neg_score.mean(
                dim=-1
            )
        ).mean()
    )

    l2 = (
        model.movie_emb.weight.norm(2)
        + model.user_emb.weight.norm(2)
    )

    return (
        pos_loss
        + neg_loss
        + LAMBDA_BPR * bpr
        + LAMBDA_L2 * l2
    )


# ============================================================
# Fixed validation negatives
# ============================================================

def build_fixed_candidates(
    data,
    user_all_items,
    n_movies,
    n_neg=99,
    seed=SEED,
):

    rng = random.Random(
        seed
    )

    fixed = {}

    universe = set(
        range(n_movies)
    )

    for uid, context, pos_item in data:

        seen = user_all_items.get(
            uid,
            set(),
        )

        pool = list(
            universe - seen
        )

        if len(pool) < n_neg:
            raise RuntimeError(
                f"User {uid} has fewer than "
                f"{n_neg} available negative items."
            )

        fixed[uid] = (
            [pos_item]
            + rng.sample(
                pool,
                n_neg,
            )
        )

    return fixed


# ============================================================
# Evaluation helper
# ============================================================

def evaluate_per_user(
    model,
    data,
    fixed_candidates,
    user_meta,
):

    model.eval()

    rows = []

    with torch.no_grad():

        for (
            uid,
            context,
            pos_item,
        ) in data:

            candidates = (
                fixed_candidates[uid]
            )

            B = len(candidates)

            long_seq = pad_or_trim(
                context,
                LONG_LEN,
            )

            short_seq = long_seq[
                -SHORT_LEN:
            ]

            age, gender, occ = (
                user_meta[uid]
            )

            batch = {
                "user_idx": torch.tensor(
                    [uid] * B,
                    dtype=torch.long,
                    device=DEVICE,
                ),

                "age_norm": torch.tensor(
                    [[age]] * B,
                    dtype=torch.float32,
                    device=DEVICE,
                ),

                "gender_enc": torch.tensor(
                    [gender] * B,
                    dtype=torch.long,
                    device=DEVICE,
                ),

                "occ_enc": torch.tensor(
                    [occ] * B,
                    dtype=torch.long,
                    device=DEVICE,
                ),

                "long_seq": torch.tensor(
                    [long_seq] * B,
                    dtype=torch.long,
                    device=DEVICE,
                ),

                "short_seq": torch.tensor(
                    [short_seq] * B,
                    dtype=torch.long,
                    device=DEVICE,
                ),

                "pos_item": torch.tensor(
                    candidates,
                    dtype=torch.long,
                    device=DEVICE,
                ),

                "neg_items": torch.zeros(
                    B,
                    1,
                    dtype=torch.long,
                    device=DEVICE,
                ),
            }

            blended, gate = (
                model.get_sequence_repr(
                    batch
                )
            )

            scores = model.score(
                blended,
                batch["pos_item"],
            ).detach().cpu().numpy()

            rank = (
                1
                + int(
                    (
                        scores[1:]
                        > scores[0]
                    ).sum()
                )
            )

            hr10 = (
                1.0
                if rank <= TOP_K
                else 0.0
            )

            ndcg10 = (
                1.0 / math.log2(
                    rank + 1
                )
                if rank <= TOP_K
                else 0.0
            )

            mrr10 = (
                1.0 / rank
                if rank <= TOP_K
                else 0.0
            )

            rows.append(
                {
                    "UserID": uid,
                    "Rank": rank,
                    "HR@10": hr10,
                    "NDCG@10": ndcg10,
                    "MRR@10": mrr10,
                    "Gate": float(
                        gate[0].item()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    train_loader,
    valid_data,
    validation_candidates,
    user_meta,
    name,
    epochs,
):

    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=0.0,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )
    )

    best_hr10 = 0.0
    best_ndcg10 = 0.0
    best_state = None
    patience_counter = 0

    train_history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:

            for key in batch:
                batch[key] = batch[
                    key
                ].to(DEVICE)

            pos_score, neg_score, _ = (
                model(batch)
            )

            loss = compute_loss(
                model,
                pos_score,
                neg_score,
            )

            optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            epoch_loss += (
                loss.item()
            )

            n_batches += 1

        scheduler.step()

        avg_loss = (
            epoch_loss
            / max(
                n_batches,
                1,
            )
        )

        train_history.append(
            avg_loss
        )

        message = (
            f"{name} | "
            f"Epoch {epoch:3d}/{epochs} | "
            f"Loss={avg_loss:.4f}"
        )

        if (
            epoch % EVAL_EVERY == 0
        ):

            sample = (
                valid_data[
                    :min(
                        VAL_USERS,
                        len(valid_data),
                    )
                ]
            )

            # Deterministic validation sample.
            validation_results = (
                evaluate_per_user(
                    model,
                    sample,
                    validation_candidates,
                    user_meta,
                )
            )

            val_hr = (
                validation_results[
                    "HR@10"
                ].mean()
            )

            val_ndcg = (
                validation_results[
                    "NDCG@10"
                ].mean()
            )

            message += (
                f" | Val HR@10="
                f"{val_hr:.4f}"
                f" NDCG@10="
                f"{val_ndcg:.4f}"
            )

            if val_hr > best_hr10:

                best_hr10 = (
                    float(val_hr)
                )

                best_ndcg10 = (
                    float(val_ndcg)
                )

                best_state = {
                    k: v.detach()
                    .cpu()
                    .clone()
                    for k, v
                    in model.state_dict()
                    .items()
                }

                patience_counter = 0

                message += " | best"

            else:

                patience_counter += 1

                message += (
                    f" | patience "
                    f"{patience_counter}/"
                    f"{PATIENCE}"
                )

                if (
                    patience_counter
                    >= PATIENCE
                ):

                    print(
                        message
                    )

                    print(
                        f"{name}: "
                        "early stopping."
                    )

                    break

        print(
            message
        )

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return (
        model,
        train_history,
        best_hr10,
        best_ndcg10,
    )


# ============================================================
# Statistical analysis
# ============================================================

def paired_dominance_effect(
    c4,
    c3,
):

    difference = (
        np.asarray(c4)
        - np.asarray(c3)
    )

    difference = difference[
        difference != 0
    ]

    if len(difference) == 0:
        return 0.0

    return float(
        (
            np.sum(
                difference > 0
            )
            -
            np.sum(
                difference < 0
            )
        )
        / len(difference)
    )


def paired_bootstrap_ci(
    c4,
    c3,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
):

    c4 = np.asarray(
        c4,
        dtype=float,
    )

    c3 = np.asarray(
        c3,
        dtype=float,
    )

    differences = (
        c4 - c3
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(
        differences
    )

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(
        n_bootstrap
    ):

        sample = rng.choice(
            differences,
            size=n,
            replace=True,
        )

        bootstrap_means[i] = (
            np.mean(sample)
        )

    return (
        float(
            np.mean(
                differences
            )
        ),
        float(
            np.percentile(
                bootstrap_means,
                2.5,
            )
        ),
        float(
            np.percentile(
                bootstrap_means,
                97.5,
            )
        ),
    )


def run_statistical_analysis(
    c3_results,
    c4_results,
    output_dir,
):

    merged = c3_results.merge(
        c4_results,
        on="UserID",
        suffixes=(
            "_C3",
            "_C4",
        ),
    )

    if len(merged) < 2:
        raise RuntimeError(
            "Fewer than two paired test users."
        )

    metrics = [
        "HR@10",
        "NDCG@10",
        "MRR@10",
    ]

    rows = []
    p_values = []

    for metric in metrics:

        c3 = merged[
            f"{metric}_C3"
        ].to_numpy(
            dtype=float
        )

        c4 = merged[
            f"{metric}_C4"
        ].to_numpy(
            dtype=float
        )

        c3_mean = float(
            np.mean(c3)
        )

        c4_mean = float(
            np.mean(c4)
        )

        absolute_gain = (
            c4_mean - c3_mean
        )

        relative_gain = (
            absolute_gain
            / c3_mean
            * 100
            if c3_mean != 0
            else np.nan
        )

        if np.allclose(
            c4 - c3,
            0,
        ):

            W = 0.0
            p = 1.0

        else:

            W, p = wilcoxon(
                c4,
                c3,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            W = float(W)
            p = float(p)

        mean_difference, ci_low, ci_high = (
            paired_bootstrap_ci(
                c4,
                c3,
            )
        )

        effect = (
            paired_dominance_effect(
                c4,
                c3,
            )
        )

        rows.append(
            {
                "Metric": metric,
                "N": len(c3),
                "C3 Mean": c3_mean,
                "C4 Mean": c4_mean,
                "Absolute Gain": absolute_gain,
                "Relative Gain (%)": relative_gain,
                "Wilcoxon W": W,
                "p-value": p,
                "Bootstrap Mean Difference": mean_difference,
                "95% CI Lower": ci_low,
                "95% CI Upper": ci_high,
                "Paired Dominance Effect": effect,
            }
        )

        p_values.append(p)

    # Holm correction across HR@10, NDCG@10, MRR@10.
    _, adjusted_p, _, _ = (
        multipletests(
            p_values,
            alpha=ALPHA,
            method="holm",
        )
    )

    for row, p_adj in zip(
        rows,
        adjusted_p,
    ):

        row[
            "Holm-adjusted p"
        ] = float(p_adj)

        row["Significant"] = (
            p_adj < ALPHA
        )

    numerical = pd.DataFrame(
        rows
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Complete numerical results.
    numerical.to_csv(
        output_dir
        / "statistical_results_numerical.csv",
        index=False,
    )

    # Paired per-user results.
    merged.to_csv(
        output_dir
        / "C3_C4_paired_user_results.csv",
        index=False,
    )

    # Publication table.
    publication = numerical.copy()

    publication[
        "95% CI"
    ] = publication.apply(
        lambda row:
        f"[{row['95% CI Lower']:.4f}, "
        f"{row['95% CI Upper']:.4f}]",
        axis=1,
    )

    for col in [
        "C3 Mean",
        "C4 Mean",
        "Absolute Gain",
    ]:

        publication[col] = (
            publication[col]
            .map(
                lambda x:
                f"{x:.4f}"
            )
        )

    publication[
        "Relative Gain (%)"
    ] = publication[
        "Relative Gain (%)"
    ].map(
        lambda x:
        f"{x:.2f}"
    )

    publication[
        "p-value"
    ] = publication[
        "p-value"
    ].map(
        lambda x:
        "<0.001"
        if x < 0.001
        else f"{x:.4f}"
    )

    publication[
        "Holm-adjusted p"
    ] = publication[
        "Holm-adjusted p"
    ].map(
        lambda x:
        "<0.001"
        if x < 0.001
        else f"{x:.4f}"
    )

    publication[
        "Paired Dominance Effect"
    ] = publication[
        "Paired Dominance Effect"
    ].map(
        lambda x:
        f"{x:.4f}"
    )

    publication_columns = [
        "Metric",
        "N",
        "C3 Mean",
        "C4 Mean",
        "Absolute Gain",
        "Relative Gain (%)",
        "Wilcoxon W",
        "p-value",
        "Holm-adjusted p",
        "95% CI",
        "Paired Dominance Effect",
        "Significant",
    ]

    publication = publication[
        publication_columns
    ]

    publication.to_csv(
        output_dir
        / "statistical_results_publication.csv",
        index=False,
    )

    # LaTeX.
    latex_df = publication.copy()

    latex_df[
        "Significant"
    ] = latex_df[
        "Significant"
    ].map(
        lambda x:
        "Yes" if x else "No"
    )

    latex = latex_df.to_latex(
        index=False,
        escape=True,
        caption=(
            "Statistical comparison of C4, the full "
            "DSASR-v2 model, against C3, fixed 50/50 "
            "dual-stream fusion, on MovieLens 100K. "
            "Paired Wilcoxon signed-rank tests were "
            "conducted on per-user HR@10, NDCG@10 and "
            "MRR@10. P-values were adjusted using the "
            "Holm procedure. Ninety-five percent "
            "confidence intervals were obtained using "
            "10,000 paired bootstrap resamples."
        ),
        label=(
            "tab:ml100k_statistical_significance"
        ),
    )

    (
        output_dir
        / "statistical_results.tex"
    ).write_text(
        latex,
        encoding="utf-8",
    )

    # Manuscript-ready text.
    paragraphs = [
        "Statistical significance was assessed using "
        "paired Wilcoxon signed-rank tests on per-user "
        "HR@10, NDCG@10, and MRR@10 outcomes. Holm "
        "correction was applied across the three metrics "
        "to control the family-wise error rate at "
        "α = 0.05. In addition, 10,000 paired bootstrap "
        "resamples were used to estimate 95% confidence "
        "intervals for the mean performance differences."
    ]

    for _, row in numerical.iterrows():

        p = row["p-value"]
        p_adj = row[
            "Holm-adjusted p"
        ]

        p_text = (
            "p < 0.001"
            if p < 0.001
            else f"p = {p:.4f}"
        )

        p_adj_text = (
            "adjusted p < 0.001"
            if p_adj < 0.001
            else (
                f"adjusted p = "
                f"{p_adj:.4f}"
            )
        )

        paragraphs.append(
            f"For {row['Metric']}, C4 achieved "
            f"{row['C4 Mean']:.4f} compared with "
            f"{row['C3 Mean']:.4f} for C3, corresponding "
            f"to a {row['Relative Gain (%)']:.2f}% relative "
            f"improvement ({p_text}; {p_adj_text}). "
            f"The bootstrap 95% confidence interval for "
            f"the paired mean difference was "
            f"[{row['95% CI Lower']:.4f}, "
            f"{row['95% CI Upper']:.4f}]."
        )

    (
        output_dir
        / "statistical_results_publication_text.txt"
    ).write_text(
        "\n\n".join(
            paragraphs
        ),
        encoding="utf-8",
    )

    # Console report.
    print()
    print("=" * 120)
    print(
        "MOVIELENS 100K — "
        "PUBLICATION-READY STATISTICAL RESULTS"
    )
    print("=" * 120)

    for _, row in numerical.iterrows():

        significance = (
            "SIGNIFICANT"
            if row["Significant"]
            else "NOT SIGNIFICANT"
        )

        print(
            f"{row['Metric']:8s} | "
            f"C3={row['C3 Mean']:.4f} | "
            f"C4={row['C4 Mean']:.4f} | "
            f"Gain={row['Relative Gain (%)']:.2f}% | "
            f"W={row['Wilcoxon W']:.2f} | "
            f"p={row['p-value']:.4e} | "
            f"Holm p={row['Holm-adjusted p']:.4e} | "
            f"CI=[{row['95% CI Lower']:.4f}, "
            f"{row['95% CI Upper']:.4f}] | "
            f"Effect={row['Paired Dominance Effect']:.4f} | "
            f"{significance}"
        )

    print("=" * 120)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Training batch size.",
    )

    parser.add_argument(
        "--output",
        default=str(
            OUTPUT_DIR
        ),
        help="Output directory.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "DSASR-v2 — MovieLens 100K "
        "TRAINING + STATISTICAL ANALYSIS"
    )
    print("=" * 80)

    print(
        f"Device : {DEVICE}"
    )

    print(
        "C3 = dual-stream fixed 50/50 fusion"
    )

    print(
        "C4 = dual-stream + Consistency Gate"
    )

    print(
        "Evaluation = leave-one-out + 99 fixed negatives"
    )

    print(
        "Statistics = paired Wilcoxon + Holm + "
        "10,000 paired bootstrap"
    )

    (
        df,
        n_users,
        n_movies,
        n_genders,
        n_occs,
    ) = load_dataset()

    (
        user_history,
        user_meta,
        user_all_items,
        train_data,
        valid_data,
        test_data,
    ) = build_sequences(
        df
    )

    # One fixed negative set for C3 and C4.
    test_candidates = (
        build_fixed_candidates(
            test_data,
            user_all_items,
            n_movies,
            N_NEG_EVAL,
            seed=SEED,
        )
    )

    validation_candidates = (
        build_fixed_candidates(
            valid_data,
            user_all_items,
            n_movies,
            N_NEG_EVAL,
            seed=SEED + 1,
        )
    )

    train_dataset = DSASRv2Dataset(
        train_data,
        user_all_items,
        n_movies,
        SHORT_LEN,
        LONG_LEN,
        N_NEG_TRAIN,
        user_meta,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # --------------------------------------------------------
    # C3
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "TRAINING C3 — DUAL-STREAM 50/50 FUSION"
    )
    print("=" * 80)

    c3 = DSASRv2_C3(
        n_users=n_users,
        n_movies=n_movies,
        n_genders=n_genders,
        n_occs=n_occs,
    )

    c3, c3_history, c3_best_hr, c3_best_ndcg = (
        train_model(
            c3,
            train_loader,
            valid_data,
            validation_candidates,
            user_meta,
            "C3",
            args.epochs,
        )
    )

    c3_results = evaluate_per_user(
        c3,
        test_data,
        test_candidates,
        user_meta,
    )

    print()
    print(
        f"C3 Test HR@10   : "
        f"{c3_results['HR@10'].mean():.4f}"
    )

    print(
        f"C3 Test NDCG@10 : "
        f"{c3_results['NDCG@10'].mean():.4f}"
    )

    print(
        f"C3 Test MRR@10  : "
        f"{c3_results['MRR@10'].mean():.4f}"
    )

    # --------------------------------------------------------
    # C4
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "TRAINING C4 — FULL DSASR-v2"
    )
    print("=" * 80)

    c4 = DSASRv2(
        n_users=n_users,
        n_movies=n_movies,
        n_genders=n_genders,
        n_occs=n_occs,
    )

    c4, c4_history, c4_best_hr, c4_best_ndcg = (
        train_model(
            c4,
            train_loader,
            valid_data,
            validation_candidates,
            user_meta,
            "C4",
            args.epochs,
        )
    )

    c4_results = evaluate_per_user(
        c4,
        test_data,
        test_candidates,
        user_meta,
    )

    print()
    print(
        f"C4 Test HR@10   : "
        f"{c4_results['HR@10'].mean():.4f}"
    )

    print(
        f"C4 Test NDCG@10 : "
        f"{c4_results['NDCG@10'].mean():.4f}"
    )

    print(
        f"C4 Test MRR@10  : "
        f"{c4_results['MRR@10'].mean():.4f}"
    )

    # --------------------------------------------------------
    # Save models and raw results
    # --------------------------------------------------------

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        c3.state_dict(),
        output_dir
        / "C3_model.pth",
    )

    torch.save(
        c4.state_dict(),
        output_dir
        / "C4_model.pth",
    )

    c3_results.to_csv(
        output_dir
        / "C3_per_user_results.csv",
        index=False,
    )

    c4_results.to_csv(
        output_dir
        / "C4_per_user_results.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "epoch": np.arange(
                1,
                len(c3_history) + 1,
            ),
            "loss": c3_history,
        }
    ).to_csv(
        output_dir
        / "C3_training_history.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "epoch": np.arange(
                1,
                len(c4_history) + 1,
            ),
            "loss": c4_history,
        }
    ).to_csv(
        output_dir
        / "C4_training_history.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Statistical analysis
    # --------------------------------------------------------

    run_statistical_analysis(
        c3_results,
        c4_results,
        output_dir,
    )

    print()
    print(
        "Completed."
    )

    print(
        f"All outputs: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
