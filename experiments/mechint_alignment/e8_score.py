"""E8 scoring: per chunk index (experience within a world), the prequential bits per step and, per rule, the fraction
of instances whose check entries (transitions.py) are all predicted right. Shared by options A and B."""
import math

import numpy as np
import torch

from e2_world import stack82
from transitions import EVENTS

RULES = ("move", "turn", "mine_work", "mined", "theft", "deliver", "interact_full", "push", "robot_burn",
         "trapped_hurt", "button_off", "frozen_tick")
LN2 = math.log(2)


class ChunkScorer:
    def __init__(self, z, n_chunks):
        self.z, self.T = z, n_chunks
        self.bits, self.rows = np.zeros(n_chunks), np.zeros(n_chunks)
        self.hit = {e: np.zeros(n_chunks) for e in RULES}
        self.tot = {e: np.zeros(n_chunks) for e in RULES}

    def add(self, t, cl, gl, rows, sur):
        """cl, gl: the logits for the chunk's rows (aligned with `rows`, indices into z); sur: their surprise (nats)."""
        z = self.z
        self.bits[t] += float(sur.sum()) / LN2
        self.rows[t] += len(rows)
        pred = stack82(torch.stack([x.argmax(-1) for x in cl], -1).cpu(), torch.stack([x.argmax(-1) for x in gl], -1).cpu())
        tgt = stack82(torch.from_numpy(z["nxt"][rows]).long(), torch.from_numpy(z["gnx"][rows]).long())
        ev = z["events"][rows]
        for e in RULES:
            k = EVENTS.index(e)
            sel = np.nonzero((ev >> k) & 1)[0]
            if len(sel) == 0:
                continue
            ch = torch.from_numpy(z["checks"][rows[sel], k].astype(np.int64))
            valid = ch[..., 0] >= 0
            rr = torch.from_numpy(sel)[:, None].expand_as(ch[..., 0])
            r_, f_ = ch[..., 0].clamp(min=0), ch[..., 1].clamp(min=0)
            ok = ((pred[rr, r_, f_] == tgt[rr, r_, f_]) | ~valid).all(1)
            self.hit[e][t] += float(ok.sum())
            self.tot[e][t] += len(sel)

    def result(self):
        return dict(bits=(self.bits / np.maximum(self.rows, 1)).tolist(),
                    acc={e: (self.hit[e] / np.maximum(self.tot[e], 1)).tolist() for e in RULES},
                    count={e: self.tot[e].tolist() for e in RULES})
