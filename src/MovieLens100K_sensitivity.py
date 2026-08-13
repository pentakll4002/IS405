
"""
DSASR-v2 — MovieLens 100K Sensitivity Analysis
================================================

Based on the supplied MovieLens 100K DSASR implementation.

This standalone script performs the additional sensitivity analyses requested
for the revised manuscript:

    4.7.1 Temporal Window Sensitivity
    4.7.2 Fusion/Gating Mechanism Sensitivity
    4.7.3 Demographic Feature Sensitivity

No hyperparameter sensitivity study is performed.

The base architecture follows the supplied implementation:
- shared movie embedding
- separate short- and long-term causal Transformer streams
- user identity + age + gender + occupation context
- Consistency Gate using cosine-based preference drift
- BCE + BPR ranking-aligned objective
- temporal leave-one-out split
- leave-one-out evaluation with 99 negative candidates

All sensitivity configurations are trained from scratch and evaluated with
the same fixed test candidate sets to make the comparison fair.

Outputs
-------
movieLens100k_sensitivity_outputs/
    ml100k_window_sensitivity.csv
    ml100k_gating_sensitivity.csv
    ml100k_demographic_sensitivity.csv
    ml100k_sensitivity_results.csv
    ml100k_sensitivity_results.tex
    ml100k_window_sensitivity.png
    ml100k_gating_sensitivity.png
    ml100k_demographic_sensitivity.png
"""

import io
import math
import random
import time
import warnings
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


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

DATASET_NAME = "MovieLens 100K"
DATASET_TAG = "ml100k"

SEED = 42
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

# Sensitivity training settings.
# These retain the same optimization family as the supplied implementation
# but avoid a large hyperparameter study.
SENS_EPOCHS = 30
SENS_EVAL_EVERY = 5
SENS_PATIENCE = 4
SENS_VAL_USERS = 200

OUTPUT_DIR = Path(
    "movieLens100k_sensitivity_outputs"
)
OUTPUT_DIR.mkdir(exist_ok=True)

warnings.filterwarnings("ignore")

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


# ============================================================================
# 2. REPRODUCIBILITY
# ============================================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=" * 78)
print("DSASR-v2 — MovieLens 100K Sensitivity Analysis")
print("=" * 78)
print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"Output : {OUTPUT_DIR.resolve()}")
print("=" * 78)


# ============================================================================
# 3. DOWNLOAD AND LOAD MOVIELENS 100K
# ============================================================================

ZIP_NAME = "ml-100k.zip"
DATA_DIR = Path("ml-100k")

if not DATA_DIR.exists():

    print("\nDownloading MovieLens 100K...")

    response = requests.get(
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        timeout=120
    )

    response.raise_for_status()

    with zipfile.ZipFile(
        io.BytesIO(response.content)
    ) as z:

        z.extractall(".")

    print("Download complete.")

ratings = pd.read_csv(
    DATA_DIR / "u.data",
    sep="\t",
    names=[
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    ]
)

users = pd.read_csv(
    DATA_DIR / "u.user",
    sep="|",
    names=[
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code"
    ]
)

GENRE_COLS = [
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
    "Western"
]

movies = pd.read_csv(
    DATA_DIR / "u.item",
    sep="|",
    encoding="latin-1",
    names=[
        "movie_id",
        "title",
        "release_date",
        "video_date",
        "imdb_url"
    ] + GENRE_COLS
)

print(f"Ratings : {len(ratings):,}")
print(f"Users   : {len(users):,}")
print(f"Movies  : {len(movies):,}")


# ============================================================================
# 4. PREPROCESSING AND DEMOGRAPHIC FEATURES
# ============================================================================

gender_le = LabelEncoder()
occ_le = LabelEncoder()

users["gender_enc"] = (
    gender_le.fit_transform(
        users["gender"]
    )
)

users["occ_enc"] = (
    occ_le.fit_transform(
        users["occupation"]
    )
)

users["age_norm"] = (
    MinMaxScaler()
    .fit_transform(
        users[["age"]]
    )
)

df = ratings.merge(
    users[
        [
            "user_id",
            "age_norm",
            "gender_enc",
            "occ_enc"
        ]
    ],
    on="user_id"
)

df = df.merge(
    movies[
        ["movie_id"] + GENRE_COLS
    ],
    on="movie_id"
)

df.sort_values(
    ["user_id", "timestamp"],
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)

user_enc = LabelEncoder()
movie_enc = LabelEncoder()

df["user_idx"] = (
    user_enc.fit_transform(
        df["user_id"]
    )
)

# Keep the same zero-based movie indexing used in the supplied implementation.
df["movie_idx"] = (
    movie_enc.fit_transform(
        df["movie_id"]
    )
)

N_USERS = df["user_idx"].nunique()
N_MOVIES = df["movie_idx"].nunique()
N_GENDERS = df["gender_enc"].nunique()
N_OCCS = df["occ_enc"].nunique()

print(f"Encoded users  : {N_USERS:,}")
print(f"Encoded movies : {N_MOVIES:,}")
print(f"Ratings        : {len(df):,}")


# ============================================================================
# 5. USER HISTORIES AND TEMPORAL SPLIT
# ============================================================================

user_history = {}
user_meta = {}

for uid, grp in df.groupby(
    "user_idx",
    sort=True
):

    grp = grp.sort_values(
        "timestamp"
    )

    user_history[uid] = (
        grp["movie_idx"]
        .astype(int)
        .tolist()
    )

    last_row = grp.iloc[-1]

    user_meta[uid] = (
        float(last_row["age_norm"]),
        int(last_row["gender_enc"]),
        int(last_row["occ_enc"])
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

    for i in range(
        1,
        len(train_hist)
    ):

        context = train_hist[
            max(0, i - LONG_LEN):i
        ]

        train_data.append(
            (
                uid,
                context,
                train_hist[i]
            )
        )

    valid_data.append(
        (
            uid,
            train_hist[-LONG_LEN:],
            hist[-2]
        )
    )

    test_data.append(
        (
            uid,
            train_hist[-LONG_LEN:],
            hist[-1]
        )
    )

print(
    f"Train : {len(train_data):,}"
)

print(
    f"Valid : {len(valid_data):,}"
)

print(
    f"Test  : {len(test_data):,}"
)


# ============================================================================
# 6. SEQUENCE DATASET
# ============================================================================

def pad_or_trim(
    seq,
    length
):

    if len(seq) >= length:
        return seq[-length:]

    return (
        [0] * (length - len(seq))
        + list(seq)
    )


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

    def _sample_negatives(
        self,
        uid,
        k
    ):

        negs = []
        seen = self.user_items.get(
            uid,
            set()
        )

        while len(negs) < k:

            neg = random.randint(
                0,
                self.n_movies - 1
            )

            if neg not in seen:
                negs.append(neg)

        return negs

    def __getitem__(
        self,
        idx
    ):

        uid, context, pos_item = (
            self.data[idx]
        )

        negs = self._sample_negatives(
            uid,
            self.n_neg
        )

        long_seq = pad_or_trim(
            context,
            self.long_len
        )

        short_seq = (
            long_seq[-self.short_len:]
        )

        age, gender, occ = (
            self.user_meta[uid]
        )

        return {
            "user_idx": torch.tensor(
                uid,
                dtype=torch.long
            ),

            "age_norm": torch.tensor(
                [age],
                dtype=torch.float32
            ),

            "gender_enc": torch.tensor(
                gender,
                dtype=torch.long
            ),

            "occ_enc": torch.tensor(
                occ,
                dtype=torch.long
            ),

            "long_seq": torch.tensor(
                long_seq,
                dtype=torch.long
            ),

            "short_seq": torch.tensor(
                short_seq,
                dtype=torch.long
            ),

            "pos_item": torch.tensor(
                pos_item,
                dtype=torch.long
            ),

            "neg_items": torch.tensor(
                negs,
                dtype=torch.long
            ),
        }


def make_train_loader(
    short_len,
    long_len
):

    dataset = DSASRv2Dataset(
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
# 7. MODEL COMPONENTS
# ============================================================================

class CausalTransformerBlock(
    nn.Module
):

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout
    ):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ff1 = nn.Linear(
            d_model,
            d_ff
        )

        self.ff2 = nn.Linear(
            d_ff,
            d_model
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
        causal=True
    ):

        B, T, _ = x.shape

        if causal:

            mask = torch.triu(
                torch.ones(
                    T,
                    T,
                    device=x.device
                ),
                diagonal=1
            ).bool()

        else:

            mask = None

        attn_out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=mask
        )

        x = self.norm1(
            x + self.drop(
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
            x + self.drop(ff)
        )

        return x


class FlexibleConsistencyGate(
    nn.Module
):

    """
    Gate modes used for the sensitivity study.

    mean:
        fixed 50/50 fusion.

    simple:
        learned gate using only short- and long-term representations.

    drift:
        learned gate using short + long + cosine-based preference drift.

    full:
        proposed gate using short + long + drift + demographic/user context.
    """

    def __init__(
        self,
        d_model,
        mode,
        ctx_dim
    ):

        super().__init__()

        self.mode = mode

        if mode == "mean":

            self.net = None

        elif mode == "simple":

            self.net = nn.Sequential(
                nn.Linear(
                    2 * d_model,
                    d_model
                ),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(
                    d_model,
                    1
                ),
                nn.Sigmoid()
            )

        elif mode == "drift":

            self.net = nn.Sequential(
                nn.Linear(
                    2 * d_model + 1,
                    d_model
                ),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(
                    d_model,
                    1
                ),
                nn.Sigmoid()
            )

        elif mode == "full":

            self.net = nn.Sequential(
                nn.Linear(
                    2 * d_model + 1 + ctx_dim,
                    d_model
                ),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(
                    d_model,
                    1
                ),
                nn.Sigmoid()
            )

        else:

            raise ValueError(
                f"Unknown gate mode: {mode}"
            )

    def forward(
        self,
        short,
        long_repr,
        user_ctx=None
    ):

        if self.mode == "mean":

            gate = torch.full(
                (
                    short.shape[0],
                    1
                ),
                0.5,
                dtype=short.dtype,
                device=short.device
            )

        elif self.mode == "simple":

            gate = self.net(
                torch.cat(
                    [
                        short,
                        long_repr
                    ],
                    dim=-1
                )
            )

        else:

            drift = (
                1.0
                - F.cosine_similarity(
                    short,
                    long_repr,
                    dim=-1,
                    eps=1e-8
                )
            ).unsqueeze(-1)

            if self.mode == "drift":

                gate_input = torch.cat(
                    [
                        short,
                        long_repr,
                        drift
                    ],
                    dim=-1
                )

            else:

                if user_ctx is None:
                    raise ValueError(
                        "Full gate requires user context."
                    )

                gate_input = torch.cat(
                    [
                        short,
                        long_repr,
                        drift,
                        user_ctx
                    ],
                    dim=-1
                )

            gate = self.net(
                gate_input
            )

        blended = (
            gate * short
            + (1.0 - gate) * long_repr
        )

        return blended, gate


class DSASRv2Flexible(
    nn.Module
):

    def __init__(
        self,
        n_users,
        n_movies,
        n_genders,
        n_occs,
        short_len,
        long_len,
        gate_mode,
        use_demographics=True
    ):

        super().__init__()

        self.short_len = short_len
        self.long_len = long_len
        self.gate_mode = gate_mode
        self.use_demographics = (
            use_demographics
        )

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

        nn.init.normal_(
            self.movie_emb.weight,
            std=0.01
        )

        nn.init.normal_(
            self.user_emb.weight,
            std=0.01
        )

        self.pos_short = nn.Embedding(
            short_len + 1,
            D_MODEL
        )

        self.pos_long = nn.Embedding(
            long_len + 1,
            D_MODEL
        )

        nn.init.normal_(
            self.pos_short.weight,
            std=0.01
        )

        nn.init.normal_(
            self.pos_long.weight,
            std=0.01
        )

        self.enc_short = nn.ModuleList(
            [
                CausalTransformerBlock(
                    D_MODEL,
                    N_HEADS,
                    D_FF,
                    DROPOUT
                )
                for _ in range(N_LAYERS)
            ]
        )

        self.enc_long = nn.ModuleList(
            [
                CausalTransformerBlock(
                    D_MODEL,
                    N_HEADS,
                    D_FF,
                    DROPOUT
                )
                for _ in range(N_LAYERS)
            ]
        )

        # The supplied implementation uses:
        # user embedding (128) + gender (8) + occupation (16) + age (1).
        ctx_dim_raw = (
            D_MODEL
            + 8
            + 16
            + 1
        )

        if use_demographics:

            self.ctx_proj = nn.Linear(
                ctx_dim_raw,
                D_MODEL
            )

        else:

            # For the demographic ablation, retain user identity but remove
            # age, gender, and occupation.
            self.ctx_proj = nn.Linear(
                D_MODEL,
                D_MODEL
            )

        gate_ctx_dim = D_MODEL

        self.gate = (
            FlexibleConsistencyGate(
                D_MODEL,
                gate_mode,
                gate_ctx_dim
            )
        )

        self.emb_dropout = nn.Dropout(
            DROPOUT
        )

        self.out_norm = nn.LayerNorm(
            D_MODEL
        )

    def encode_sequence(
        self,
        seq_ids,
        pos_emb,
        encoder_blocks
    ):

        T = seq_ids.shape[1]

        movie_emb = self.movie_emb(
            seq_ids
        )

        pos = torch.arange(
            T,
            device=seq_ids.device
        ).unsqueeze(0)

        x = self.emb_dropout(
            movie_emb
            + pos_emb(pos)
        )

        for block in encoder_blocks:

            x = block(
                x,
                causal=True
            )

        return x

    def get_sequence_repr(
        self,
        batch
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
            self.enc_long
        )

        short_enc = self.encode_sequence(
            short_seq,
            self.pos_short,
            self.enc_short
        )

        long_repr = (
            long_enc[:, -1, :]
        )

        short_repr = (
            short_enc[:, -1, :]
        )

        user_ctx = None

        if self.gate_mode == "full":

            u_idx = batch[
                "user_idx"
            ].to(dev)

            u_emb = self.user_emb(
                u_idx
            )

            if self.use_demographics:

                g_idx = batch[
                    "gender_enc"
                ].to(dev)

                o_idx = batch[
                    "occ_enc"
                ].to(dev)

                age = batch[
                    "age_norm"
                ].to(dev)

                g_emb = self.gender_emb(
                    g_idx
                )

                o_emb = self.occ_emb(
                    o_idx
                )

                raw_ctx = torch.cat(
                    [
                        u_emb,
                        g_emb,
                        o_emb,
                        age
                    ],
                    dim=-1
                )

            else:

                raw_ctx = u_emb

            user_ctx = self.ctx_proj(
                raw_ctx
            )

        blended, gate_val = (
            self.gate(
                short_repr,
                long_repr,
                user_ctx
            )
        )

        blended = self.out_norm(
            blended
        )

        return blended, gate_val

    def score(
        self,
        seq_repr,
        item_ids
    ):

        item_emb = self.movie_emb(
            item_ids
        )

        if item_ids.dim() == 1:

            return (
                seq_repr * item_emb
            ).sum(dim=-1)

        return torch.bmm(
            item_emb,
            seq_repr.unsqueeze(-1)
        ).squeeze(-1)

    def forward(
        self,
        batch
    ):

        blended, gate_val = (
            self.get_sequence_repr(
                batch
            )
        )

        pos_score = self.score(
            blended,
            batch["pos_item"].to(
                next(self.parameters()).device
            )
        )

        neg_score = self.score(
            blended,
            batch["neg_items"].to(
                next(self.parameters()).device
            )
        )

        return (
            pos_score,
            neg_score,
            gate_val
        )


# ============================================================================
# 8. FIXED EVALUATION CANDIDATES
# ============================================================================

def build_eval_candidates(
    data,
    seed
):

    rng = random.Random(seed)

    candidates = []

    all_items = set(
        range(N_MOVIES)
    )

    for uid, context, pos_item in data:

        seen = user_all_items.get(
            uid,
            set()
        )

        pool = list(
            all_items - seen
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
            (
                uid,
                context,
                pos_item,
                negatives
            )
        )

    return candidates


# The exact same candidate sets are used by every sensitivity configuration.
VALID_CANDIDATES = build_eval_candidates(
    valid_data,
    seed=12345
)

TEST_CANDIDATES = build_eval_candidates(
    test_data,
    seed=54321
)

print(
    f"Fixed validation candidates: "
    f"{len(VALID_CANDIDATES):,}"
)

print(
    f"Fixed test candidates: "
    f"{len(TEST_CANDIDATES):,}"
)


# ============================================================================
# 9. EVALUATION
# ============================================================================

def evaluate_model(
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

        for (
            uid,
            context,
            pos_item,
            negatives
        ) in candidates:

            candidate_items = (
                [pos_item]
                + negatives
            )

            B = len(
                candidate_items
            )

            long_seq = pad_or_trim(
                context,
                long_len
            )

            short_seq = (
                long_seq[-short_len:]
            )

            age, gender, occ = (
                user_meta[uid]
            )

            batch = {

                "user_idx":
                    torch.tensor(
                        [uid] * B,
                        dtype=torch.long,
                        device=DEVICE
                    ),

                "age_norm":
                    torch.tensor(
                        [[age]] * B,
                        dtype=torch.float32,
                        device=DEVICE
                    ),

                "gender_enc":
                    torch.tensor(
                        [gender] * B,
                        dtype=torch.long,
                        device=DEVICE
                    ),

                "occ_enc":
                    torch.tensor(
                        [occ] * B,
                        dtype=torch.long,
                        device=DEVICE
                    ),

                "long_seq":
                    torch.tensor(
                        [long_seq] * B,
                        dtype=torch.long,
                        device=DEVICE
                    ),

                "short_seq":
                    torch.tensor(
                        [short_seq] * B,
                        dtype=torch.long,
                        device=DEVICE
                    ),

                "pos_item":
                    torch.tensor(
                        candidate_items,
                        dtype=torch.long,
                        device=DEVICE
                    ),

                "neg_items":
                    torch.zeros(
                        B,
                        1,
                        dtype=torch.long,
                        device=DEVICE
                    )
            }

            representation, _ = (
                model.get_sequence_repr(
                    batch
                )
            )

            scores = model.score(
                representation,
                batch["pos_item"]
            )

            scores = (
                scores
                .detach()
                .cpu()
                .numpy()
            )

            rank = (
                1
                + int(
                    (
                        scores[1:]
                        > scores[0]
                    ).sum()
                )
            )

            hr.append(
                1.0
                if rank <= 10
                else 0.0
            )

            ndcg.append(
                1.0 / math.log2(
                    rank + 1
                )
                if rank <= 10
                else 0.0
            )

            mrr.append(
                1.0 / rank
                if rank <= 10
                else 0.0
            )

    return {
        "HR@10": float(
            np.mean(hr)
        ),
        "NDCG@10": float(
            np.mean(ndcg)
        ),
        "MRR@10": float(
            np.mean(mrr)
        )
    }


# ============================================================================
# 10. TRAINING
# ============================================================================

def train_configuration(
    label,
    short_len,
    long_len,
    gate_mode,
    use_demographics,
    seed
):

    print("\n" + "=" * 78)
    print(
        f"TRAINING: {label}"
    )
    print("=" * 78)

    print(
        f"short={short_len}, "
        f"long={long_len}, "
        f"gate={gate_mode}, "
        f"demographics={use_demographics}"
    )

    set_seed(seed)

    loader = make_train_loader(
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

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=SENS_EPOCHS,
            eta_min=1e-5
        )
    )

    best_hr = -1.0
    best_state = None
    patience_count = 0

    start = time.time()

    for epoch in range(
        1,
        SENS_EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0
        n_batches = 0

        for batch in loader:

            batch = {
                k: v.to(DEVICE)
                for k, v in batch.items()
            }

            pos_score, neg_score, _ = (
                model(batch)
            )

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
                model.movie_emb.weight.norm(
                    2
                )
                + model.user_emb.weight.norm(
                    2
                )
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

            total_loss += (
                loss.item()
            )

            n_batches += 1

        scheduler.step()

        if (
            epoch % SENS_EVAL_EVERY
            == 0
        ):

            validation_subset = (
                VALID_CANDIDATES[
                    :min(
                        SENS_VAL_USERS,
                        len(
                            VALID_CANDIDATES
                        )
                    )
                ]
            )

            val = evaluate_model(
                model,
                validation_subset,
                short_len,
                long_len
            )

            avg_loss = (
                total_loss
                / max(
                    n_batches,
                    1
                )
            )

            print(
                f"Epoch {epoch:02d}/"
                f"{SENS_EPOCHS} | "
                f"Loss={avg_loss:.4f} | "
                f"Val HR@10="
                f"{val['HR@10']:.4f}"
            )

            if (
                val["HR@10"]
                > best_hr + 1e-6
            ):

                best_hr = (
                    val["HR@10"]
                )

                best_state = {
                    k:
                        v.detach()
                        .cpu()
                        .clone()

                    for k, v
                    in model.state_dict().items()
                }

                patience_count = 0

            else:

                patience_count += 1

                if (
                    patience_count
                    >= SENS_PATIENCE
                ):

                    print(
                        "Early stopping."
                    )

                    break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    test = evaluate_model(
        model,
        TEST_CANDIDATES,
        short_len,
        long_len
    )

    elapsed = (
        time.time() - start
    ) / 60.0

    print(
        f"FINAL {label}: "
        f"HR@10={test['HR@10']:.4f}, "
        f"NDCG@10={test['NDCG@10']:.4f}, "
        f"MRR@10={test['MRR@10']:.4f}, "
        f"time={elapsed:.1f} min"
    )

    return {
        "label": label,
        "short_len": short_len,
        "long_len": long_len,
        "gate_mode": gate_mode,
        "demographics": (
            "Yes"
            if use_demographics
            else "No"
        ),
        "HR@10": test["HR@10"],
        "NDCG@10": test["NDCG@10"],
        "MRR@10": test["MRR@10"],
        "training_minutes": elapsed
    }


# ============================================================================
# 11. 4.7.1 TEMPORAL WINDOW SENSITIVITY
# ============================================================================

print("\n")
print("#" * 78)
print(
    "4.7.1 TEMPORAL WINDOW SENSITIVITY — MOVIELENS 100K"
)
print("#" * 78)

WINDOW_CONFIGS = [

    (
        "W1: 5/20",
        5,
        20
    ),

    (
        "W2: 5/50",
        5,
        50
    ),

    (
        "W3: 10/50 (default)",
        10,
        50
    ),

    (
        "W4: 10/100",
        10,
        100
    ),

    (
        "W5: 20/100",
        20,
        100
    )
]

window_results = []

for i, (
    label,
    short_len,
    long_len
) in enumerate(
    WINDOW_CONFIGS
):

    row = train_configuration(
        label=label,
        short_len=short_len,
        long_len=long_len,
        gate_mode="full",
        use_demographics=True,
        seed=SEED + 100 + i
    )

    row["study"] = (
        "Temporal Window Sensitivity"
    )

    window_results.append(
        row
    )

window_df = pd.DataFrame(
    window_results
)

window_df.to_csv(
    OUTPUT_DIR
    / "ml100k_window_sensitivity.csv",
    index=False
)


# ============================================================================
# 12. 4.7.2 FUSION/GATING MECHANISM SENSITIVITY
# ============================================================================

print("\n")
print("#" * 78)
print(
    "4.7.2 FUSION/GATING MECHANISM SENSITIVITY — MOVIELENS 100K"
)
print("#" * 78)

GATE_CONFIGS = [

    (
        "G1: Fixed 50/50 Mean Fusion",
        "mean"
    ),

    (
        "G2: Learned Gate (short + long)",
        "simple"
    ),

    (
        "G3: Drift-Aware Gate",
        "drift"
    ),

    (
        "G4: Full Consistency Gate (proposed)",
        "full"
    )
]

gate_results = []

for i, (
    label,
    gate_mode
) in enumerate(
    GATE_CONFIGS
):

    row = train_configuration(
        label=label,
        short_len=SHORT_LEN,
        long_len=LONG_LEN,
        gate_mode=gate_mode,
        use_demographics=True,
        seed=SEED + 200 + i
    )

    row["study"] = (
        "Fusion/Gating Sensitivity"
    )

    gate_results.append(
        row
    )

gate_df = pd.DataFrame(
    gate_results
)

gate_df.to_csv(
    OUTPUT_DIR
    / "ml100k_gating_sensitivity.csv",
    index=False
)


# ============================================================================
# 13. 4.7.3 DEMOGRAPHIC FEATURE SENSITIVITY
# ============================================================================

print("\n")
print("#" * 78)
print(
    "4.7.3 DEMOGRAPHIC FEATURE SENSITIVITY — MOVIELENS 100K"
)
print("#" * 78)

DEMO_CONFIGS = [

    (
        "D1: Gate without demographics",
        False
    ),

    (
        "D2: Full Consistency Gate + demographics",
        True
    )
]

demo_results = []

for i, (
    label,
    use_demographics
) in enumerate(
    DEMO_CONFIGS
):

    row = train_configuration(
        label=label,
        short_len=SHORT_LEN,
        long_len=LONG_LEN,
        gate_mode="full",
        use_demographics=use_demographics,
        seed=SEED + 300 + i
    )

    row["study"] = (
        "Demographic Feature Sensitivity"
    )

    demo_results.append(
        row
    )

demo_df = pd.DataFrame(
    demo_results
)

demo_df.to_csv(
    OUTPUT_DIR
    / "ml100k_demographic_sensitivity.csv",
    index=False
)


# ============================================================================
# 14. COMBINE RESULTS
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
    OUTPUT_DIR
    / "ml100k_sensitivity_results.csv",
    index=False
)


# ============================================================================
# 15. RELATIVE PERFORMANCE CHANGE
# ============================================================================

def add_relative_gain(
    df,
    reference_label
):

    df = df.copy()

    ref = df[
        df["label"]
        == reference_label
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

        if base == 0:

            df[
                metric + "_gain_%"
            ] = np.nan

        else:

            df[
                metric + "_gain_%"
            ] = (
                (
                    df[metric]
                    - base
                )
                / base
                * 100.0
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
# 16. LATEX TABLES
# ============================================================================

def make_latex_table(
    df,
    caption,
    label
):

    lines = [

        r"\begin{table}[!htb]",

        r"\centering",

        r"\caption{"
        + caption
        + "}",

        r"\label{"
        + label
        + "}",

        r"\begin{tabular}{lccc}",

        r"\toprule",

        (
            r"Configuration & HR@10 & "
            r"NDCG@10 & MRR@10 \\"
        ),

        r"\midrule"
    ]

    for _, row in df.iterrows():

        name = (
            str(row["label"])
            .replace(
                "_",
                r"\_"
            )
            .replace(
                "&",
                r"\&"
            )
        )

        lines.append(
            f"{name} & "
            f"{row['HR@10']:.4f} & "
            f"{row['NDCG@10']:.4f} & "
            f"{row['MRR@10']:.4f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ]
    )

    return "\n".join(
        lines
    )


latex = "\n\n".join(
    [

        make_latex_table(
            window_df,

            (
                "Temporal window sensitivity "
                "of DSASR on MovieLens 100K."
            ),

            "tab:ml100k_window_sensitivity"
        ),

        make_latex_table(
            gate_df,

            (
                "Fusion and gating mechanism "
                "sensitivity of DSASR on MovieLens 100K."
            ),

            "tab:ml100k_gating_sensitivity"
        ),

        make_latex_table(
            demo_df,

            (
                "Effect of demographic features "
                "in the Consistency Gate on MovieLens 100K."
            ),

            "tab:ml100k_demographic_sensitivity"
        )
    ]
)

(
    OUTPUT_DIR
    / "ml100k_sensitivity_results.tex"
).write_text(
    latex,
    encoding="utf-8"
)


# ============================================================================
# 17. PUBLICATION-READY PLOTS
# ============================================================================

def plot_sensitivity(
    df,
    title,
    filename
):

    metrics = [
        "HR@10",
        "NDCG@10",
        "MRR@10"
    ]

    x = np.arange(
        len(df)
    )

    width = 0.24

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    for j, metric in enumerate(
        metrics
    ):

        bars = ax.bar(
            x
            + (j - 1) * width,

            df[metric].values,

            width,

            label=metric
        )

        for bar in bars:

            value = (
                bar.get_height()
            )

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

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
            str(v).replace(
                ":",
                ":\n",
                1
            )
            if len(str(v)) > 24
            else str(v)

            for v in df["label"]
        ],

        rotation=25,
        ha="right"
    )

    ax.set_ylabel(
        "Ranking Score"
    )

    ax.set_title(
        title,
        fontweight="bold"
    )

    ymax = float(
        df[metrics]
        .max()
        .max()
    )

    ax.set_ylim(
        0,
        max(
            1.0,
            ymax * 1.20
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

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: "
        f"{OUTPUT_DIR / filename}"
    )


plot_sensitivity(
    window_df,
    (
        "Temporal Window Sensitivity "
        "— MovieLens 100K"
    ),
    "ml100k_window_sensitivity.png"
)

plot_sensitivity(
    gate_df,
    (
        "Fusion/Gating Sensitivity "
        "— MovieLens 100K"
    ),
    "ml100k_gating_sensitivity.png"
)

plot_sensitivity(
    demo_df,
    (
        "Demographic Feature Sensitivity "
        "— MovieLens 100K"
    ),
    "ml100k_demographic_sensitivity.png"
)


# ============================================================================
# 18. FINAL SUMMARY
# ============================================================================

print("\n")
print("=" * 78)
print(
    "MOVIELENS 100K SENSITIVITY ANALYSIS COMPLETE"
)
print("=" * 78)

print(
    "\n[4.7.1] Temporal Window Sensitivity"
)

print(
    window_df[
        [
            "label",
            "HR@10",
            "NDCG@10",
            "MRR@10"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\n[4.7.2] Fusion/Gating Sensitivity"
)

print(
    gate_df[
        [
            "label",
            "HR@10",
            "NDCG@10",
            "MRR@10"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\n[4.7.3] Demographic Feature Sensitivity"
)

print(
    demo_df[
        [
            "label",
            "HR@10",
            "NDCG@10",
            "MRR@10"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nOutput files:"
)

for f in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        f"  {f}"
    )

print(
    "\nNo hyperparameter sensitivity study "
    "was performed."
)
