import warnings, random, math, time, sys, os, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# Determinism flags — reduces GPU non-determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f'Device : {DEVICE}')
print(f'PyTorch: {torch.__version__}')

import zipfile
DATASET_NAME = 'MovieLens 1M'
DATASET_TAG  = 'ml1m'

import requests, io, zipfile

print('Downloading MovieLens 1M ...')
r = requests.get('https://files.grouplens.org/datasets/movielens/ml-1m.zip')
zipfile.ZipFile(io.BytesIO(r.content)).extractall('.')
print('Done.')

def bce_loss(pos_score, neg_score):
    """
    Binary Cross-Entropy with negative sampling.
    pos_score: (B,)    — score for the ground-truth next item
    neg_score: (B, K)  — scores for K negative items

    Loss = -log(sigmoid(pos)) - mean_K(log(1 - sigmoid(neg)))
    This is exactly the SASRec training objective.
    """
    pos_loss = -F.logsigmoid(pos_score).mean()
    neg_loss = -F.logsigmoid(-neg_score).mean()
    return pos_loss + neg_loss


def bpr_loss(pos_score, neg_score):
    """
    Bayesian Personalised Ranking loss.
    Encourages pos_score > mean(neg_score).
    Complementary to BCE — adds pairwise ranking signal.
    """
    mean_neg = neg_score.mean(dim=-1)   # (B,)
    return -F.logsigmoid(pos_score - mean_neg).mean()


LAMBDA_BPR = 0.5   # weight on BPR term
LAMBDA_L2  = 1e-4  # L2 regularisation weight
# Ratings: UserID::MovieID::Rating::Timestamp
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                      names=['user_id','movie_id','rating','timestamp'])

# Users: UserID::Gender::Age::Occupation::Zip
users   = pd.read_csv('ml-1m/users.dat', sep='::', engine='python',
                      names=['user_id','gender','age','occupation','zip'])

# Movies: MovieID::Title::Genres
movies  = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python',
                      encoding='latin-1',
                      names=['movie_id','title','genres'])

print(f'Ratings : {len(ratings):,}')
print(f'Users   : {len(users):,}')
print(f'Movies  : {len(movies):,}')


# Encode user demographics
gender_le = LabelEncoder(); occ_le = LabelEncoder()
users['gender_enc'] = gender_le.fit_transform(users['gender'])
users['occ_enc']    = occ_le.fit_transform(users['occupation'])
users['age_norm']   = MinMaxScaler().fit_transform(users[['age']])

# Merge ratings with user features
df = ratings.merge(users[['user_id','age_norm','gender_enc','occ_enc']], on='user_id')
df.sort_values(['user_id','timestamp'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Encode user and movie indices
user_enc  = LabelEncoder(); movie_enc = LabelEncoder()
df['user_idx']  = user_enc.fit_transform(df['user_id'])
df['movie_idx'] = movie_enc.fit_transform(df['movie_id'])

N_USERS   = df['user_idx'].nunique()
N_MOVIES  = df['movie_idx'].nunique()
N_GENDERS = df['gender_enc'].nunique()
N_OCCS    = df['occ_enc'].nunique()

print(f'Users  : {N_USERS:,}')
print(f'Movies : {N_MOVIES:,}')
print(f'Ratings: {len(df):,}')
print(f'Avg interactions/user: {len(df)/N_USERS:.1f}')
print(f'Sparsity: {1 - len(df)/(N_USERS*N_MOVIES):.4%}')



# ── Hyperparameters ───────────────────────────────────────────────────────────
SHORT_LEN   = 10    # recent window
LONG_LEN    = 50    # historical window
MIN_HIST    = 5     # skip users with fewer interactions
N_NEG_TRAIN = 5     # negatives per positive during training
BATCH        = 256
D_MODEL      = 128
LR           = 1e-3
LAMBDA_BPR   = 0.5
EVAL_EVERY   = 5

print(f'Sequence windows: short={SHORT_LEN}, long={LONG_LEN}')
print(f'Embedding dim   : {D_MODEL}')
print()
print('Expected training time:')
print('  Colab T4 GPU : ~25-35 min (50 epochs)')
print('  CPU only     : ~5-7 hrs')

# ── Build per-user histories ──────────────────────────────────────────────────
user_history = {}
user_meta    = {}   # uid -> (age_norm, gender_enc, occ_enc)

for uid, grp in df.groupby('user_idx', sort=True):
    user_history[uid] = grp.sort_values('timestamp')['movie_idx'].tolist()
    r = grp.iloc[-1]
    user_meta[uid] = (float(r['age_norm']), int(r['gender_enc']), int(r['occ_enc']))

user_all_items = {uid: set(h) for uid, h in user_history.items()}

# ── Temporal leave-one-out split ──────────────────────────────────────────────
train_data, valid_data, test_data = [], [], []

for uid, hist in user_history.items():
    if len(hist) < MIN_HIST: continue
    train_hist = hist[:-2]
    for i in range(1, len(train_hist)):
        ctx = train_hist[max(0, i-LONG_LEN): i]
        train_data.append((uid, ctx, train_hist[i]))
    ctx_vt = train_hist[-LONG_LEN:]
    valid_data.append((uid, ctx_vt, hist[-2]))
    test_data.append( (uid, ctx_vt, hist[-1]))

print(f'\nTrain: {len(train_data):,} | Valid: {len(valid_data):,} | Test: {len(test_data):,}')



def pad_or_trim(seq, length):
    """Pad with 0 or trim to exact length."""
    if len(seq) >= length:
        return seq[-length:]
    return [0] * (length - len(seq)) + seq


class DSASRv2Dataset(Dataset):
    """
    Generic dataset for all three corpora (MovieLens, Beauty).
    No genre_lookup required — genre features were removed from DSASR-v2
    to keep the architecture corpus-agnostic.
    Each sample yields: padded long/short sequences + pos item + N_NEG_TRAIN negatives.
    """
    def __init__(self, data, user_all_items, n_movies,
                 short_len, long_len, n_neg, user_meta):
        self.data       = data
        self.user_items = user_all_items
        self.n_movies   = n_movies
        self.short_len  = short_len
        self.long_len   = long_len
        self.n_neg      = n_neg
        self.user_meta  = user_meta

    def __len__(self):
        return len(self.data)

    def _sample_negatives(self, uid, k):
        negs = []
        seen = self.user_items.get(uid, set())
        while len(negs) < k:
            neg = random.randint(0, self.n_movies - 1)
            if neg not in seen:
                negs.append(neg)
        return negs

    def __getitem__(self, idx):
        uid, context, pos_item = self.data[idx]
        negs      = self._sample_negatives(uid, self.n_neg)
        long_seq  = pad_or_trim(context, self.long_len)
        short_seq = long_seq[-self.short_len:]
        age, gender, occ = self.user_meta[uid]
        return {
            'user_idx'  : torch.tensor(uid,       dtype=torch.long),
            'age_norm'  : torch.tensor([age],     dtype=torch.float32),
            'gender_enc': torch.tensor(gender,    dtype=torch.long),
            'occ_enc'   : torch.tensor(occ,       dtype=torch.long),
            'long_seq'  : torch.tensor(long_seq,  dtype=torch.long),
            'short_seq' : torch.tensor(short_seq, dtype=torch.long),
            'pos_item'  : torch.tensor(pos_item,  dtype=torch.long),
            'neg_items' : torch.tensor(negs,      dtype=torch.long),
        }


train_set    = DSASRv2Dataset(train_data, user_all_items, N_MOVIES,
                              SHORT_LEN, LONG_LEN, N_NEG_TRAIN, user_meta)
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                          num_workers=0, drop_last=False)
print(f'Train loader ready: {len(train_loader)} batches')



class CausalTransformerBlock(nn.Module):
    """
    Transformer block with CAUSAL (left-to-right) masking.
    Position i can only attend to positions 0..i — no future leakage.
    Same as SASRec.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.ff1   = nn.Linear(d_model, d_ff)
        self.ff2   = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, causal=True):
        B, T, _ = x.shape
        if causal:
            # Upper-triangular mask: positions can only see past
            mask = torch.triu(
                torch.ones(T, T, device=x.device), diagonal=1
            ).bool()
        else:
            mask = None
        attn_out, attn_weights = self.attn(
            x, x, x,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=False
        )

        self.last_attn = attn_weights.detach()
        x = self.norm1(x + self.drop(attn_out))
        ff = self.ff2(self.drop(F.gelu(self.ff1(x))))
        x  = self.norm2(x + self.drop(ff))
        return x


class ConsistencyGate(nn.Module):
    """
    Novel contribution: learns a per-user, per-prediction blend weight g
    between the short-term and long-term sequence representations.
    drift = 1 - cosine_similarity(short, long)
    g = sigmoid(MLP([short || long || drift || user_ctx]))
    blended = g * short + (1-g) * long
    """
    def __init__(self, d_model, ctx_dim, hidden=64):
        super().__init__()
        gate_in = d_model * 2 + 1 + ctx_dim
        self.net  = nn.Sequential(
            nn.Linear(gate_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, short, long_, user_ctx):
        drift   = (1.0 - F.cosine_similarity(short, long_, dim=-1, eps=1e-8)).unsqueeze(-1)
        gate_in = torch.cat([short, long_, drift, user_ctx], dim=-1)
        g       = self.net(gate_in)                   # (B, 1)
        blended = g * short + (1 - g) * long_
        return blended, g


class DSASRv2(nn.Module):
    """
    DSASR-v2: Dual-Stream Adaptive Sequential Recommendation

    Architecture:
      1. Movie embedding (shared table, d=128)
      2. Learnable positional encoding per stream
      3. Short-term transformer (causal, 2 blocks, uses last SHORT_LEN items)
      4. Long-term transformer (causal, 2 blocks, uses last LONG_LEN items)
      5. Consistency Gate: learns drift-aware blend of short and long repr
      6. User context (user_emb + gender + occ + age) injected into gate
      7. Scoring: dot product of blended repr with candidate item embedding
         (same mechanism as SASRec — directly optimises ranking)

    Training loss:
      L = BCE(pos_score, 1) + BCE(neg_score, 0)  [for each negative]
      + lambda * BPR(pos_score - neg_score mean)
    """
    def __init__(self, n_users, n_movies, n_genders, n_occs,
                 d_model=128, n_heads=4, d_ff=256, n_layers=2,
                 short_len=10, long_len=50, dropout=0.2):
        super().__init__()
        self.d_model   = d_model
        self.short_len = short_len
        self.long_len  = long_len

        # ── Embedding tables ───────────────────────────────────────────────
        # n_movies+1 because 0 is padding index
        self.movie_emb  = nn.Embedding(n_movies + 1, d_model, padding_idx=0)
        self.user_emb   = nn.Embedding(n_users  + 1, d_model)
        self.gender_emb = nn.Embedding(n_genders + 1, 8)
        self.occ_emb    = nn.Embedding(n_occs   + 1, 16)

        # Initialise movie embeddings with small normal (helps convergence)
        nn.init.normal_(self.movie_emb.weight, std=0.01)
        nn.init.normal_(self.user_emb.weight,  std=0.01)

        # ── Positional encodings (separate per stream) ─────────────────────
        self.pos_short = nn.Embedding(short_len + 1, d_model)
        self.pos_long  = nn.Embedding(long_len  + 1, d_model)
        nn.init.normal_(self.pos_short.weight, std=0.01)
        nn.init.normal_(self.pos_long.weight,  std=0.01)

        # ── Transformer encoders ───────────────────────────────────────────
        # Short: CAUSAL (matches SASRec design, prevents future leakage)
        self.enc_short = nn.ModuleList(
            [CausalTransformerBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)])
        # Long: bidirectional (sees full history context)
        self.enc_long  = nn.ModuleList(
            [CausalTransformerBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)])

        # ── User context ───────────────────────────────────────────────────
        # user_emb(128) + gender(8) + occ(16) + age(1) = 153
        ctx_dim = d_model + 8 + 16 + 1
        self.ctx_proj = nn.Linear(ctx_dim, d_model)   # project to d_model for gate

        # ── Consistency Gate ───────────────────────────────────────────────
        self.gate = ConsistencyGate(d_model, d_model, hidden=d_model)

        # ── Layer norms & dropout ─────────────────────────────────────────
        self.emb_dropout = nn.Dropout(dropout)
        self.out_norm    = nn.LayerNorm(d_model)

    def encode_sequence(self, seq_ids, pos_emb, encoder_blocks, causal=True):
        """
        Encode a padded sequence of item ids.
        seq_ids: (B, T)  — 0 = padding
        Returns: (B, T, d_model)  — last position is the summary used
        """
        B, T    = seq_ids.shape
        m_emb   = self.movie_emb(seq_ids)              # (B, T, d)
        pos     = torch.arange(T, device=seq_ids.device).unsqueeze(0)  # (1, T)
        p_emb   = pos_emb(pos)                         # (1, T, d)
        x       = self.emb_dropout(m_emb + p_emb)      # (B, T, d)
        for block in encoder_blocks:
            x = block(x, causal=causal)
        return x

    def get_sequence_repr(self, batch):
        """
        Returns blended repr (B, d) and gate value g (B, 1).
        """
        dev      = next(self.parameters()).device
        long_seq = batch['long_seq'].to(dev)    # (B, LONG_LEN)
        short_seq= batch['short_seq'].to(dev)   # (B, SHORT_LEN)
        u_idx    = batch['user_idx'].to(dev)
        g_idx    = batch['gender_enc'].to(dev)
        o_idx    = batch['occ_enc'].to(dev)
        age      = batch['age_norm'].to(dev)

        # Encode both streams; take last non-padding position
        long_enc  = self.encode_sequence(long_seq,  self.pos_long,  self.enc_long,  causal=True)
        short_enc = self.encode_sequence(short_seq, self.pos_short, self.enc_short, causal=True)

        # Take the LAST position as the sequence representation
        # (SASRec style: position -1 predicts the next item)
        long_repr  = long_enc[:, -1, :]    # (B, d)
        short_repr = short_enc[:, -1, :]   # (B, d)

        # User context
        u_emb    = self.user_emb(u_idx)     # (B, d)
        g_emb    = self.gender_emb(g_idx)   # (B, 8)
        o_emb    = self.occ_emb(o_idx)      # (B, 16)
        user_ctx = torch.cat([u_emb, g_emb, o_emb, age], dim=-1)  # (B, 153)
        user_ctx_proj = self.ctx_proj(user_ctx)  # (B, d)

        # Consistency Gate
        blended, gate_val = self.gate(short_repr, long_repr, user_ctx_proj)
        blended = self.out_norm(blended)    # (B, d)

        return blended, gate_val

    def score(self, seq_repr, item_ids):
        """
        Score items via dot product (SASRec mechanism).
        seq_repr: (B, d)
        item_ids: (B,) or (B, K)
        Returns: (B,) or (B, K)
        """
        if item_ids.dim() == 1:
            item_emb = self.movie_emb(item_ids)          # (B, d)
            return (seq_repr * item_emb).sum(dim=-1)      # (B,)
        else:
            item_emb = self.movie_emb(item_ids)          # (B, K, d)
            return torch.bmm(item_emb, seq_repr.unsqueeze(-1)).squeeze(-1)  # (B, K)

    def forward(self, batch):
        blended, gate_val = self.get_sequence_repr(batch)
        dev = next(self.parameters()).device
        pos_score = self.score(blended, batch['pos_item'].to(dev))
        neg_score = self.score(blended, batch['neg_items'].to(dev))  # (B, K)
        return pos_score, neg_score, gate_val


# Instantiate
D_MODEL = 128
N_LAYERS = 2

model = DSASRv2(
    n_users  = N_USERS,
    n_movies = N_MOVIES,
    n_genders= N_GENDERS,
    n_occs   = N_OCCS,
    d_model  = D_MODEL,
    n_heads  = 4,
    d_ff     = D_MODEL * 4,
    n_layers = N_LAYERS,
    short_len= SHORT_LEN,
    long_len = LONG_LEN,
    dropout  = 0.2,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'DSASR-v2 parameters: {total_params:,}')



model = DSASRv2(
    n_users   = N_USERS,
    n_movies  = N_MOVIES,
    n_genders = N_GENDERS,
    n_occs    = N_OCCS,
    d_model   = D_MODEL,
    n_heads   = 4,
    d_ff      = D_MODEL * 4,
    n_layers  = 2,
    short_len = SHORT_LEN,
    long_len  = LONG_LEN,
    dropout   = 0.2,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'DSASR-v2 (MovieLens 1M) parameters: {total_params:,}')




import time
import sys

# ── Batched validation — much faster than per-user loop ───────────────────
def quick_eval_batched(model, valid_data, user_all_items, n_movies,
                       n_sample=200, n_neg=99):
    """
    Evaluate HR@10 and NDCG@10 on a sample of validation users.
    Batches all candidates together for a single forward pass per user.
    ~10x faster than the per-user loop version.
    """
    model.eval()
    if len(valid_data) > n_sample:
        sample = random.sample(valid_data, n_sample)
    else:
        sample = valid_data

    hr10, ndcg10 = [], []

    with torch.no_grad():
        for uid, context, pos_item in sample:
            neg_pool = list(set(range(n_movies)) - user_all_items.get(uid, set()))
            negs     = random.sample(neg_pool, min(n_neg, len(neg_pool)))
            cands    = [pos_item] + negs
            B        = len(cands)

            long_seq  = pad_or_trim(context, LONG_LEN)
            short_seq = long_seq[-SHORT_LEN:]
            age, gender, occ = user_meta[uid]

            # Single forward pass for all B candidates at once
            batch = {
                'user_idx'  : torch.tensor([uid]*B,        dtype=torch.long,    device=DEVICE),
                'age_norm'  : torch.tensor([[age]]*B,       dtype=torch.float32, device=DEVICE),
                'gender_enc': torch.tensor([gender]*B,      dtype=torch.long,    device=DEVICE),
                'occ_enc'   : torch.tensor([occ]*B,         dtype=torch.long,    device=DEVICE),
                'long_seq'  : torch.tensor([long_seq]*B,    dtype=torch.long,    device=DEVICE),
                'short_seq' : torch.tensor([short_seq]*B,   dtype=torch.long,    device=DEVICE),
                'pos_item'  : torch.tensor(cands,           dtype=torch.long,    device=DEVICE),
                'neg_items' : torch.zeros(B, 1,             dtype=torch.long,    device=DEVICE),
            }

            blended, _ = model.get_sequence_repr(batch)
            scores = model.score(blended, batch['pos_item']).cpu().numpy()

            rank = 1 + int((scores[1:] > scores[0]).sum())
            hr10.append(1.0 if rank <= 10 else 0.0)
            ndcg10.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)

    model.train()
    return float(np.mean(hr10)), float(np.mean(ndcg10))


# ── Training configuration ────────────────────────────────────────────────
EPOCHS      = 50      # 50 is enough with cosine schedule + early stopping
LR          = 1e-3
PATIENCE    = 5       # stop if no improvement for 5 consecutive val checks
EVAL_EVERY  = 5       # validate every N epochs
LAMBDA_BPR  = 0.5
LAMBDA_L2   = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5)

train_losses  = []
val_hr10_hist = []
val_nd10_hist = []
best_hr10     = 0.0
best_nd10     = 0.0
best_state    = None
pat_counter   = 0

print(f"Training DSASR-v2 for up to {EPOCHS} epochs...")
print(f"Validating every {EVAL_EVERY} epochs on {min(200,len(valid_data))} users")
print(f"Device: {DEVICE}")
print("-" * 55)
sys.stdout.flush()

t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for batch in train_loader:
        # Move batch to device
        for k in batch:
            batch[k] = batch[k].to(DEVICE)

        pos_score, neg_score, gate_val = model(batch)

        # BCE + BPR loss
        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).mean()
        bpr      = -F.logsigmoid(pos_score - neg_score.mean(dim=-1)).mean()
        l2       = (model.movie_emb.weight.norm(2) +
                    model.user_emb.weight.norm(2))
        loss = pos_loss + neg_loss + LAMBDA_BPR * bpr + LAMBDA_L2 * l2

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    avg_loss = epoch_loss / max(n_batches, 1)
    train_losses.append(avg_loss)

    elapsed = (time.time() - t0) / 60
    print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
          f"Elapsed: {elapsed:.1f}m", end="")
    sys.stdout.flush()

    if epoch % EVAL_EVERY == 0:
        hr10, nd10 = quick_eval_batched(
            model, valid_data, user_all_items, N_MOVIES, n_sample=200)
        val_hr10_hist.append(hr10)
        val_nd10_hist.append(nd10)
        print(f" | Val HR@10: {hr10:.4f}  NDCG@10: {nd10:.4f}", end="")
        sys.stdout.flush()

        if hr10 > best_hr10:
            best_hr10   = hr10
            best_nd10   = nd10
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            pat_counter = 0
            print(" ✓ best", end="")
        else:
            pat_counter += 1
            if pat_counter >= PATIENCE:
                print(f"\nEarly stopping — no improvement for {PATIENCE} checks.")
                break
    print()
    sys.stdout.flush()

# Restore best model
if best_state:
    model.load_state_dict(best_state)

total_min = (time.time() - t0) / 60
print(f"\nDone in {total_min:.1f} min | Best Val HR@10={best_hr10:.4f} NDCG@10={best_nd10:.4f}")




# ════════════════════════════════════════════════════════════════════════════
# FINAL TEST EVALUATION — FIXED CANDIDATES FOR REPRODUCIBLE STATISTICS
#
# The same 99 negative candidates are used for C1, C2, C3 and C4 for each
# test user. This makes the model comparisons paired at the user level and
# avoids changing the evaluation candidates between configurations.
# ════════════════════════════════════════════════════════════════════════════

NS         = (5, 10, 20)
N_NEG_EVAL = 99


def build_fixed_candidates(data, n_movies, user_all_items, n_neg=99, seed=42):
    """Create one fixed candidate list per evaluation instance.

    The first item is always the ground-truth item and the remaining items
    are sampled negatives. The resulting candidate sets are reused by every
    model configuration in the ablation study.
    """
    rng = random.Random(seed)
    candidates = {}

    for idx, (uid, context, pos_item) in enumerate(data):
        neg_pool = list(set(range(n_movies)) - user_all_items.get(uid, set()))
        neg_count = min(n_neg, len(neg_pool))
        negs = rng.sample(neg_pool, neg_count)
        candidates[idx] = [pos_item] + negs

    return candidates


def make_eval_batch(uid, context, candidates):
    """Construct a ranking-evaluation batch for one user."""
    B = len(candidates)
    long_seq = pad_or_trim(context, LONG_LEN)
    short_seq = long_seq[-SHORT_LEN:]
    age, gender, occ = user_meta[uid]

    return {
        'user_idx': torch.tensor([uid] * B,
                                 dtype=torch.long, device=DEVICE),
        'age_norm': torch.tensor([[age]] * B,
                                 dtype=torch.float32, device=DEVICE),
        'gender_enc': torch.tensor([gender] * B,
                                    dtype=torch.long, device=DEVICE),
        'occ_enc': torch.tensor([occ] * B,
                                dtype=torch.long, device=DEVICE),
        'long_seq': torch.tensor([long_seq] * B,
                                 dtype=torch.long, device=DEVICE),
        'short_seq': torch.tensor([short_seq] * B,
                                  dtype=torch.long, device=DEVICE),
        'pos_item': torch.tensor(candidates,
                                 dtype=torch.long, device=DEVICE),
        'neg_items': torch.zeros(B, N_NEG_TRAIN,
                                 dtype=torch.long, device=DEVICE),
    }


def evaluate_model_fixed_candidates(model, data, fixed_candidates,
                                    ks=(5, 10, 20)):
    """Return per-user HR, NDCG and MRR using fixed candidate sets."""
    model.eval()
    scores_by_k = {
        'HR': {k: [] for k in ks},
        'NDCG': {k: [] for k in ks},
        'MRR': {k: [] for k in ks},
        'rank': []
    }

    with torch.no_grad():
        for idx, (uid, context, pos_item) in enumerate(data):
            candidates = fixed_candidates[idx]
            batch = make_eval_batch(uid, context, candidates)
            blended, _ = model.get_sequence_repr(batch)
            scores = model.score(
                blended, batch['pos_item']
            ).detach().cpu().numpy()

            # Positive item is at index 0.
            rank = 1 + int((scores[1:] > scores[0]).sum())
            scores_by_k['rank'].append(rank)

            for k in ks:
                scores_by_k['HR'][k].append(
                    1.0 if rank <= k else 0.0
                )
                scores_by_k['NDCG'][k].append(
                    1.0 / math.log2(rank + 1) if rank <= k else 0.0
                )
                scores_by_k['MRR'][k].append(
                    1.0 / rank if rank <= k else 0.0
                )

    return {
        'HR': {k: np.asarray(v, dtype=float)
               for k, v in scores_by_k['HR'].items()},
        'NDCG': {k: np.asarray(v, dtype=float)
                 for k, v in scores_by_k['NDCG'].items()},
        'MRR': {k: np.asarray(v, dtype=float)
                for k, v in scores_by_k['MRR'].items()},
        'rank': np.asarray(scores_by_k['rank'], dtype=int)
    }


# Generate the common test candidate sets once.
fixed_test_candidates = build_fixed_candidates(
    test_data,
    N_MOVIES,
    user_all_items,
    n_neg=N_NEG_EVAL,
    seed=SEED + 1000
)

# C4 = the fully trained DSASR-v2 model.
c4_eval = evaluate_model_fixed_candidates(
    model,
    test_data,
    fixed_test_candidates,
    ks=NS
)

hr_scores   = c4_eval['HR']
ndcg_scores = c4_eval['NDCG']
mrr_scores  = c4_eval['MRR']

hr_mean   = {n: float(np.mean(hr_scores[n])) for n in NS}
ndcg_mean = {n: float(np.mean(ndcg_scores[n])) for n in NS}
mrr_mean  = {n: float(np.mean(mrr_scores[n])) for n in NS}

print('=' * 78)
print(f'       DSASR-v2 — {DATASET_NAME}')
print('       Protocol: leave-one-out + fixed 99 negative candidates')
print(f'       Test users: {len(test_data):,}')
print('=' * 78)
for n in NS:
    print(
        f'  HR@{n:<2}: {hr_mean[n]:.4f}   '
        f'NDCG@{n:<2}: {ndcg_mean[n]:.4f}   '
        f'MRR@{n:<2}: {mrr_mean[n]:.4f}'
    )
print('=' * 78)


# ════════════════════════════════════════════════════════════════════════════
# CELL 11 — ABLATION STUDY + PUBLICATION-READY STATISTICAL ANALYSIS
#


# ============================================================
# STANDALONE C3 TRAINING + STATISTICAL TESTING
# ============================================================
# C3 = Dual-stream with fixed 50/50 fusion + BCE+BPR.
# C4 = Full DSASR-v2 with Consistency Gate + BCE+BPR.
#
# This script trains ONLY the two configurations required for the
# statistical comparison, saves their checkpoints, evaluates them on
# IDENTICAL per-user test candidates, and performs publication-ready
# paired statistical tests.
# ============================================================

import json
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

STAT_OUTPUT_DIR = 'dsasr_statistical_outputs'
os.makedirs(STAT_OUTPUT_DIR, exist_ok=True)

C4_CHECKPOINT = os.path.join(STAT_OUTPUT_DIR, 'DSASRv2_C4_checkpoint.pth')
C3_CHECKPOINT = os.path.join(STAT_OUTPUT_DIR, 'DSASRv2_C3_checkpoint.pth')

# ------------------------------------------------------------
# Save the already-trained C4 model from the main training above
# ------------------------------------------------------------
torch.save(
    {
        'model_state_dict': model.state_dict(),
        'configuration': 'C4: DSASR-v2 + Consistency Gate + BCE+BPR',
        'dataset': DATASET_NAME,
        'seed': SEED,
    },
    C4_CHECKPOINT,
)
print(f'\nSaved C4 checkpoint: {C4_CHECKPOINT}')


class DSASRv2_C3(DSASRv2):
    """C3: dual-stream representation with fixed 50/50 fusion."""
    def get_sequence_repr(self, batch):
        dev = next(self.parameters()).device
        long_seq = batch['long_seq'].to(dev)
        short_seq = batch['short_seq'].to(dev)

        long_enc = self.encode_sequence(
            long_seq, self.pos_long, self.enc_long, causal=True
        )
        short_enc = self.encode_sequence(
            short_seq, self.pos_short, self.enc_short, causal=True
        )

        long_repr = long_enc[:, -1, :]
        short_repr = short_enc[:, -1, :]

        blended = self.out_norm(
            0.5 * short_repr + 0.5 * long_repr
        )

        gate = torch.full(
            (blended.shape[0], 1),
            0.5,
            device=dev,
        )

        return blended, gate


# ------------------------------------------------------------
# Common fixed validation candidates for C3 model selection
# ------------------------------------------------------------
ABLATION_MAX_EPOCHS = 50
ABLATION_EVAL_EVERY = 5
ABLATION_PATIENCE = 5
ABLATION_VAL_USERS = 200
ABLATION_N_NEG = 99

val_rng = random.Random(SEED + 2000)
fixed_val_indices = val_rng.sample(
    range(len(valid_data)),
    min(ABLATION_VAL_USERS, len(valid_data))
)

fixed_val_candidates = build_fixed_candidates(
    valid_data,
    N_MOVIES,
    user_all_items,
    n_neg=ABLATION_N_NEG,
    seed=SEED + 3000,
)


def evaluate_validation_hr10(model):
    model.eval()
    values = []

    with torch.no_grad():
        for idx in fixed_val_indices:
            uid, context, pos_item = valid_data[idx]
            candidates = fixed_val_candidates[idx]
            batch = make_eval_batch(uid, context, candidates)
            blended, _ = model.get_sequence_repr(batch)
            scores = model.score(
                blended, batch['pos_item']
            ).detach().cpu().numpy()

            rank = 1 + int((scores[1:] > scores[0]).sum())
            values.append(1.0 if rank <= 10 else 0.0)

    return float(np.mean(values))


def train_c3():
    print('\n' + '=' * 78)
    print('TRAINING C3 — Dual-stream fixed 50/50 fusion + BCE+BPR')
    print('=' * 78)

    m = DSASRv2_C3(
        n_users=N_USERS,
        n_movies=N_MOVIES,
        n_genders=N_GENDERS,
        n_occs=N_OCCS,
        d_model=D_MODEL,
        n_heads=4,
        d_ff=D_MODEL * 4,
        n_layers=2,
        short_len=SHORT_LEN,
        long_len=LONG_LEN,
        dropout=0.2,
    ).to(DEVICE)

    opt = torch.optim.Adam(
        m.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    best_val = -np.inf
    best_state = None
    patience = 0

    for epoch in range(1, ABLATION_MAX_EPOCHS + 1):
        m.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)

            ps, ns, _ = m(batch)

            loss = bce_loss(ps, ns)
            loss = loss + LAMBDA_BPR * (
                -F.logsigmoid(
                    ps - ns.mean(-1)
                ).mean()
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                m.parameters(),
                5.0,
            )
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % ABLATION_EVAL_EVERY == 0:
            val_hr = evaluate_validation_hr10(m)

            print(
                f'C3 Epoch {epoch:3d}/{ABLATION_MAX_EPOCHS} | '
                f'Loss={avg_loss:.4f} | '
                f'Val HR@10={val_hr:.4f}'
            )

            if val_hr > best_val + 1e-5:
                best_val = val_hr
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in m.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= ABLATION_PATIENCE:
                    print('C3 early stopping.')
                    break

    if best_state is not None:
        m.load_state_dict(best_state)

    torch.save(
        {
            'model_state_dict': m.state_dict(),
            'configuration': 'C3: dual-stream fixed 50/50 fusion + BCE+BPR',
            'dataset': DATASET_NAME,
            'seed': SEED,
            'best_validation_hr10': best_val,
        },
        C3_CHECKPOINT,
    )

    print(f'Saved C3 checkpoint: {C3_CHECKPOINT}')
    return m


# ------------------------------------------------------------
# Train C3 only. C4 has already been trained above.
# ------------------------------------------------------------
c3_model = train_c3()


# ------------------------------------------------------------
# Fixed test candidates: SAME candidates for C3 and C4
# ------------------------------------------------------------
fixed_test_candidates = build_fixed_candidates(
    test_data,
    N_MOVIES,
    user_all_items,
    n_neg=N_NEG_EVAL,
    seed=SEED + 1000,
)


# ------------------------------------------------------------
# Per-user evaluation
# ------------------------------------------------------------
def evaluate_per_user(model):
    model.eval()

    records = []

    with torch.no_grad():
        for idx, (uid, context, pos_item) in enumerate(test_data):
            candidates = fixed_test_candidates[idx]
            batch = make_eval_batch(
                uid,
                context,
                candidates,
            )

            blended, gate = model.get_sequence_repr(batch)

            scores = model.score(
                blended,
                batch['pos_item'],
            ).detach().cpu().numpy()

            rank = 1 + int(
                (scores[1:] > scores[0]).sum()
            )

            records.append({
                'UserID': uid,
                'Rank': rank,
                'HR@10': float(rank <= 10),
                'NDCG@10': (
                    float(1.0 / math.log2(rank + 1))
                    if rank <= 10 else 0.0
                ),
                'MRR@10': (
                    float(1.0 / rank)
                    if rank <= 10 else 0.0
                ),
                'Gate': float(gate[0].item()),
            })

    return pd.DataFrame(records)


print('\nEvaluating C3 on fixed test candidates...')
c3_eval = evaluate_per_user(c3_model)

print('Evaluating C4 on fixed test candidates...')
c4_eval = evaluate_per_user(model)

# Save raw per-user outcomes.
c3_eval.to_csv(
    os.path.join(STAT_OUTPUT_DIR, 'C3_per_user_results.csv'),
    index=False,
)
c4_eval.to_csv(
    os.path.join(STAT_OUTPUT_DIR, 'C4_per_user_results.csv'),
    index=False,
)

paired = c3_eval.merge(
    c4_eval,
    on='UserID',
    suffixes=('_C3', '_C4'),
)

paired.to_csv(
    os.path.join(STAT_OUTPUT_DIR, 'C3_C4_paired_results.csv'),
    index=False,
)

print(f'Paired test users: {len(paired):,}')


# ============================================================
# Publication-ready statistical analysis
# ============================================================
def paired_dominance_effect(c4, c3):
    diff = np.asarray(c4) - np.asarray(c3)
    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    return float(
        (np.sum(diff > 0) - np.sum(diff < 0)) / len(diff)
    )


def bootstrap_ci(c4, c3, n_boot=10000, seed=42):
    diff = np.asarray(c4, dtype=float) - np.asarray(c3, dtype=float)
    rng = np.random.default_rng(seed)

    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(diff, size=len(diff), replace=True)
        boot[i] = np.mean(sample)

    return (
        float(np.mean(diff)),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


metrics = [
    ('HR@10', 'HR@10_C3', 'HR@10_C4'),
    ('NDCG@10', 'NDCG@10_C3', 'NDCG@10_C4'),
    ('MRR@10', 'MRR@10_C3', 'MRR@10_C4'),
]

rows = []
raw_p = []

for metric, c3_col, c4_col in metrics:
    c3 = paired[c3_col].to_numpy(float)
    c4 = paired[c4_col].to_numpy(float)

    diff = c4 - c3

    if np.allclose(diff, 0):
        w = 0.0
        p = 1.0
    else:
        w, p = wilcoxon(
            c4,
            c3,
            alternative='two-sided',
            zero_method='wilcox',
            method='auto',
        )
        w = float(w)
        p = float(p)

    mean_diff, ci_low, ci_high = bootstrap_ci(
        c4,
        c3,
        n_boot=10000,
        seed=SEED,
    )

    c3_mean = float(np.mean(c3))
    c4_mean = float(np.mean(c4))

    rel_gain = (
        100.0 * (c4_mean - c3_mean) / c3_mean
        if c3_mean != 0 else np.nan
    )

    delta = paired_dominance_effect(c4, c3)

    rows.append({
        'Metric': metric,
        'N': len(c3),
        'C3 Mean': c3_mean,
        'C4 Mean': c4_mean,
        'Absolute Gain': mean_diff,
        'Relative Gain (%)': rel_gain,
        'Wilcoxon W': w,
        'p-value': p,
        'Bootstrap 95% CI Lower': ci_low,
        'Bootstrap 95% CI Upper': ci_high,
        'Paired Dominance Effect': delta,
    })

    raw_p.append(p)

_, holm_p, _, _ = multipletests(
    raw_p,
    alpha=0.05,
    method='holm',
)

for row, p_adj in zip(rows, holm_p):
    row['Holm-adjusted p'] = float(p_adj)
    row['Significant'] = bool(p_adj < 0.05)

stats_df = pd.DataFrame(rows)

# ------------------------------------------------------------
# Numerical results
# ------------------------------------------------------------
stats_df.to_csv(
    os.path.join(
        STAT_OUTPUT_DIR,
        'statistical_results_numerical.csv',
    ),
    index=False,
)

# ------------------------------------------------------------
# Publication table
# ------------------------------------------------------------
pub = stats_df.copy()

pub['C3 Mean'] = pub['C3 Mean'].map(lambda x: f'{x:.4f}')
pub['C4 Mean'] = pub['C4 Mean'].map(lambda x: f'{x:.4f}')
pub['Absolute Gain'] = pub['Absolute Gain'].map(lambda x: f'{x:.4f}')
pub['Relative Gain (%)'] = pub['Relative Gain (%)'].map(lambda x: f'{x:.2f}')
pub['Wilcoxon W'] = pub['Wilcoxon W'].map(lambda x: f'{x:.2f}')
pub['p-value'] = pub['p-value'].map(
    lambda x: '<0.001' if x < 0.001 else f'{x:.4f}'
)
pub['Holm-adjusted p'] = pub['Holm-adjusted p'].map(
    lambda x: '<0.001' if x < 0.001 else f'{x:.4f}'
)
pub['95% CI'] = pub.apply(
    lambda r: (
        f'[{r["Bootstrap 95% CI Lower"]:.4f}, '
        f'{r["Bootstrap 95% CI Upper"]:.4f}]'
    ),
    axis=1,
)
pub['Paired Dominance Effect'] = pub[
    'Paired Dominance Effect'
].map(lambda x: f'{x:.4f}')
pub['Significant'] = pub['Significant'].map(
    lambda x: 'Yes' if x else 'No'
)

pub = pub[
    [
        'Metric', 'N', 'C3 Mean', 'C4 Mean',
        'Absolute Gain', 'Relative Gain (%)',
        'Wilcoxon W', 'p-value', 'Holm-adjusted p',
        '95% CI', 'Paired Dominance Effect', 'Significant'
    ]
]

pub_csv = os.path.join(
    STAT_OUTPUT_DIR,
    'statistical_results_publication.csv',
)
pub.to_csv(pub_csv, index=False)

# ------------------------------------------------------------
# LaTeX table
# ------------------------------------------------------------
latex = pub.to_latex(
    index=False,
    escape=True,
    caption=(
        'Statistical comparison of C3 (dual-stream fixed 50/50 fusion) '
        'and C4 (DSASR-v2 with the Consistency Gate) on MovieLens 1M. '
        'Paired Wilcoxon signed-rank tests are conducted on per-user '
        'recommendation outcomes. P-values are Holm-adjusted across '
        'the three metrics. Confidence intervals are obtained from '
        '10,000 paired bootstrap resamples.'
    ),
    label='tab:dsasr_statistical_significance',
)

tex_path = os.path.join(
    STAT_OUTPUT_DIR,
    'statistical_results.tex',
)
Path(tex_path).write_text(
    latex,
    encoding='utf-8',
)

# ------------------------------------------------------------
# Publication-ready manuscript paragraph
# ------------------------------------------------------------
paragraph = [
    'Statistical significance was assessed using paired Wilcoxon '
    'signed-rank tests on per-user HR@10, NDCG@10, and MRR@10 '
    'outcomes. Holm correction was applied across the three '
    'metrics to control the family-wise error rate at α = 0.05. '
    'In addition, 10,000 paired bootstrap resamples were used to '
    'estimate 95% confidence intervals for the mean performance '
    'differences.'
]

for _, r in stats_df.iterrows():
    p_text = '<0.001' if r['p-value'] < 0.001 else f"{r['p-value']:.4f}"
    hp_text = '<0.001' if r['Holm-adjusted p'] < 0.001 else f"{r['Holm-adjusted p']:.4f}"

    paragraph.append(
        f"For {r['Metric']}, C4 achieved {r['C4 Mean']:.4f} "
        f"compared with {r['C3 Mean']:.4f} for C3, corresponding "
        f"to a {r['Relative Gain (%)']:.2f}% relative improvement. "
        f"The paired Wilcoxon test yielded p = {p_text} "
        f"(Holm-adjusted p = {hp_text}), with a bootstrap 95% "
        f"confidence interval of [{r['Bootstrap 95% CI Lower']:.4f}, "
        f"{r['Bootstrap 95% CI Upper']:.4f}] for the paired mean "
        f"difference."
    )

text_path = os.path.join(
    STAT_OUTPUT_DIR,
    'statistical_results_publication_text.txt',
)
Path(text_path).write_text(
    '\n\n'.join(paragraph),
    encoding='utf-8',
)

# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------
print('\n' + '=' * 115)
print('PUBLICATION-READY STATISTICAL RESULTS — C3 vs C4')
print('=' * 115)

for _, r in stats_df.iterrows():
    print(
        f"{r['Metric']:8s} | "
        f"C3={r['C3 Mean']:.4f} | "
        f"C4={r['C4 Mean']:.4f} | "
        f"Gain={r['Relative Gain (%)']:.2f}% | "
        f"W={r['Wilcoxon W']:.2f} | "
        f"p={r['p-value']:.4e} | "
        f"Holm p={r['Holm-adjusted p']:.4e} | "
        f"95% CI=[{r['Bootstrap 95% CI Lower']:.4f}, "
        f"{r['Bootstrap 95% CI Upper']:.4f}] | "
        f"Effect={r['Paired Dominance Effect']:.4f} | "
        f"{'SIGNIFICANT' if r['Significant'] else 'NOT SIGNIFICANT'}"
    )

print('=' * 115)
print('Outputs:')
print(f'  C3 checkpoint: {C3_CHECKPOINT}')
print(f'  C4 checkpoint: {C4_CHECKPOINT}')
print(f'  Paired users : {len(paired):,}')
print(f'  Results CSV  : {pub_csv}')
print(f'  LaTeX table  : {tex_path}')
print(f'  Text section : {text_path}')
print('=' * 115)
