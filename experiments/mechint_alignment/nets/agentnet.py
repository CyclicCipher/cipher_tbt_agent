"""The robot's network for E3: ONE trunk (worldmodel.WorldModel's, without the action input) and three heads.
- policy and value: read the global token;
- world model: predicts the next observation for the action taken. Each cell token's trunk features, plus the
  global token's and the action's embedding, go through a small per-token MLP, then the field heads with the
  persistence prior. There is no attention after the trunk, so the trunk itself must encode what each cell's
  neighbourhood will do under each action: the refutations shape the same features the policy reads.
"""
import torch
import torch.nn as nn

from nets.worldmodel import WorldModel


class AgentNet(WorldModel):
    def __init__(self, spec, d=128, layers=4, heads=4, mlp=4, pos="rel", persist=5.0):
        super().__init__(spec, d, layers, heads, mlp, pos=pos, persist=persist)
        self.pi, self.v = nn.Linear(d, spec.n_actions), nn.Linear(d, 1)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.wm_act = nn.Embedding(spec.n_actions, d)
        self.wm_mix = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def trunk(self, cells, glob):
        return self.features(cells, glob, None)

    def forward(self, cells, glob):
        """-> (action logits [B, 8], value [B]) in fp32; the Roller and the evaluators call this."""
        h = self.trunk(cells, glob)
        with torch.autocast(h.device.type, enabled=False):
            return self.pi(h[:, 0]), self.v(h[:, 0]).squeeze(-1)

    def policy_and_features(self, cells, glob):
        h = self.trunk(cells, glob)
        with torch.autocast(h.device.type, enabled=False):
            return self.pi(h[:, 0]), self.v(h[:, 0]).squeeze(-1), h

    def predict(self, h, cells, glob, act):
        """The world-model head on trunk features h for actions act -> (cell logits, global logits)."""
        hc, hg = h[:, self.interior], h[:, 0]
        ctx = hg + self.wm_act(act.long())
        zc = hc + self.wm_mix(torch.cat([hc, ctx[:, None].expand_as(hc)], -1))
        zg = hg + self.wm_mix(torch.cat([hg, ctx], -1))
        with torch.autocast(h.device.type, enabled=False):
            cl = [m(zc.float()) for m in self.cell_heads]
            gl = [m(zg.float()) for m in self.glob_heads]
            if self.persist is not None:
                nc = len(cl)
                oh = torch.nn.functional.one_hot
                cl = [x + self.persist[f] * oh(cells[..., f].long(), x.shape[-1]) for f, x in enumerate(cl)]
                gl = [x + self.persist[nc + f] * oh(glob[..., f].long(), x.shape[-1]) for f, x in enumerate(gl)]
        return cl, gl
