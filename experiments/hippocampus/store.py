"""The hippocampal store: a table of episodes, indexed by sparse binary codes (PLAN.md §2).

Every token id has a fixed, seeded, random sparse binary code (D bits, k active). An episode's index is the UNION of its
tokens' codes, an unordered set, so a cue matches an episode equally whichever of its tokens the cue shares: retrieval
is direction-free by construction. A cue's overlap with an episode is the number of shared active bits.

`index="dense"` is H1's control: each token gets a fixed random N(0, 1) vector, an episode's key is the sum (also
order-free) and similarity is cosine. It has no sparsity, hence no pattern separation.

Rows are the episode's tokens plus bookkeeping (write surprise, evidence count, last use). The codes are a function of
the token ids, so a row's minimal size is its tokens plus metadata; `self.keys` is a cache for fast overlap.
"""
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Store:
    def __init__(self, vocab, index="sparse", D=2048, k=40, d_dense=128, seed=12345, device=DEV):
        g = torch.Generator().manual_seed(seed)
        self.index, self.device = index, device
        if index == "sparse":
            codes = torch.zeros(vocab, D)
            for v in range(vocab):
                codes[v, torch.randperm(D, generator=g)[:k]] = 1.0
        else:
            codes = torch.randn(vocab, d_dense, generator=g)
        self.codes = codes.to(device)
        self.rows = torch.zeros(0, 0, dtype=torch.long, device=device)
        self.keys = torch.zeros(0, self.codes.shape[1], device=device)
        self.surprise, self.evidence, self.last = [], [], []
        self.where = {}                                        # episode tokens -> row ids (for diagnostics)
        self.step = 0
        self.expansion = None                                  # H2b: (vocab,) learned partner token per token, or -1
        self.read_order = "ranked"                             # H1c: "ranked" (best last) or "shuffled"

    def __len__(self):
        return self.rows.shape[0]

    def key(self, toks):
        c = self.codes[toks]                                   # (n, L, D)
        return c.amax(1) if self.index == "sparse" else c.sum(1)

    def write(self, toks, surprise=None):
        """Append episodes (n, L) as new rows: one-shot, never overwriting."""
        toks = toks.to(self.device)
        for i, t in enumerate(toks.tolist()):
            self.where.setdefault(tuple(t), []).append(len(self) + i)
        self.rows = toks if len(self) == 0 else torch.cat([self.rows, toks])
        self.keys = torch.cat([self.keys, self.key(toks)])
        n = toks.shape[0]
        self.surprise += [None] * n if surprise is None else [float(s) for s in surprise]
        self.evidence += [0] * n
        self.last += [self.step] * n
        self.born = getattr(self, "born", []) + [self.step] * n

    def remove(self, ids):
        """Evict rows (H2): rebuild the table without them."""
        drop = set(int(i) for i in ids)
        keep = [i for i in range(len(self)) if i not in drop]
        k = torch.tensor(keep, dtype=torch.long, device=self.device)
        self.rows, self.keys = self.rows[k], self.keys[k]
        for name in ("surprise", "evidence", "last", "born"):
            setattr(self, name, [getattr(self, name)[i] for i in keep])
        self.where = {}
        for i, t in enumerate(self.rows.tolist()):
            self.where.setdefault(tuple(t), []).append(i)

    def familiar(self, toks):
        """Recognition: True where an identical episode is already stored. A candidate is any row whose index contains
        the whole of the episode's code (pattern completion); it is then verified against the row's tokens, because a
        set index cannot tell (a, r, b) from (b, r, a)."""
        toks = toks.to(self.device)
        if len(self) == 0:
            return [False] * len(toks)
        q = self.key(toks)
        full = (q @ self.keys.T) >= q.sum(1, keepdim=True) - 0.5      # (n, N): the episode's code is contained
        out = []
        for i in range(len(toks)):
            cand = full[i].nonzero().flatten()
            out.append(bool(cand.numel()) and bool((self.rows[cand] == toks[i]).all(1).any()))
        return out

    def retrieve(self, qtoks, R, theta=20.0, gen=None):
        """Row ids (n, R) of the best-matching episodes, best first; -1 where fewer than R clear the threshold."""
        if len(self) == 0:
            return torch.full((qtoks.shape[0], R), -1, dtype=torch.long, device=self.device)
        q = self.key(qtoks.to(self.device))
        if self.expansion is not None:                          # add the learned partners' codes to the query
            ex = self.expansion.to(self.device)[qtoks.to(self.device)]
            c = self.codes[ex.clamp(min=0)] * (ex >= 0)[..., None]
            q = torch.maximum(q, c.amax(1)) if self.index == "sparse" else q + c.sum(1)
        if self.index == "sparse":
            s = q @ self.keys.T                                 # exact overlap counts
            s = s + 0.5 * torch.rand(s.shape, device=self.device, generator=gen)   # random tie-break, < 1 bit
            top = s.topk(min(R, len(self)), dim=1)
            ids = torch.where(top.values >= theta, top.indices, torch.full_like(top.indices, -1))
        else:
            s = torch.nn.functional.normalize(q, dim=1) @ torch.nn.functional.normalize(self.keys, dim=1).T
            ids = s.topk(min(R, len(self)), dim=1).indices
        if ids.shape[1] < R:
            ids = torch.cat([ids, torch.full((ids.shape[0], R - ids.shape[1]), -1, device=self.device,
                                             dtype=torch.long)], 1)
        return ids

    def row_bytes(self):
        """Minimal bytes per row: tokens (2 bytes each) + surprise (fp16) + evidence (uint16) + last use (uint32)."""
        return 2 * self.rows.shape[1] + 2 + 2 + 4
