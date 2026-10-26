"""The one network (E6): the robot's world model, reward, values and policy all in ONE set of weights.

Trunk: worldmodel.WorldModel's, without the action input (relative-position attention, the locality window, the
persistence prior on the next-state outputs). Then, per action a, one representation of the global token:
    z_a = h_glob + mix([h_glob, h_glob + E_a])
which feeds
- the next global fields (the world model's global part), and for cells the same mix with each cell's features;
- reward(s, a): the reward of taking a here;
- Q(s, a): the expected return of taking a here -- the robot's VALUES, in the weights.
The policy is a softmax over Q / temperature. Q and reward heads start at zero, so a fresh network babbles uniformly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.worldmodel import WorldModel


class OneNet(WorldModel):
    def __init__(self, spec, d=64, layers=3, heads=4, mlp=4, pos="local", persist=5.0, radius=1):
        super().__init__(spec, d, layers, heads, mlp, pos=pos, persist=persist, radius=radius)
        self.act_emb = nn.Embedding(spec.n_actions, d)
        self.mix = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.q, self.r = nn.Linear(d, 1), nn.Linear(d, 1)
        for m in (self.q, self.r):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def trunk(self, cells, glob):
        return self.features(cells, glob, None)

    def per_action(self, h):
        """z [B, A, d]: the global token's representation of each action's consequence."""
        hg = h[:, 0].float()
        ctx = hg[:, None] + self.act_emb.weight[None]
        return hg[:, None] + self.mix(torch.cat([hg[:, None].expand_as(ctx), ctx], -1))

    def values(self, h):
        """Q [B, A] and reward [B, A] for every action, from one trunk pass."""
        z = self.per_action(h)
        return self.q(z).squeeze(-1), self.r(z).squeeze(-1)

    def predict_next(self, h, cells, glob, act):
        """The world-model outputs for the actions taken: (cell logits, global logits), with the persistence prior."""
        hc, hg = h[:, self.interior].float(), h[:, 0].float()
        ctx = hg + self.act_emb(act.long())
        zc = hc + self.mix(torch.cat([hc, ctx[:, None].expand_as(hc)], -1))
        zg = self.per_action(h)[torch.arange(len(act), device=h.device), act.long()]
        cl = [m(zc) for m in self.cell_heads]
        gl = [m(zg) for m in self.glob_heads]
        if self.persist is not None:
            nc = len(cl)
            cl = [x + self.persist[f] * F.one_hot(cells[..., f].long(), x.shape[-1]) for f, x in enumerate(cl)]
            gl = [x + self.persist[nc + f] * F.one_hot(glob[..., f].long(), x.shape[-1]) for f, x in enumerate(gl)]
        return cl, gl
