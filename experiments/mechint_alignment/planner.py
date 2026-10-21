"""Acting by planning in the learned world model (E4): no policy learning at all.

- RewardModel: a linear model of the reward on GENERIC features of the global token: one-hot of each field's change and
  of its new value. Least squares on logged experience; no hand-chosen features.
- Planner: breadth-first search over action sequences in the world model's own argmax predictions, de-duplicated by
  the robot-relevant part of the predicted state (its position and facing, carried gold, health, freeze, and every cell's
  terrain and amount). It returns the shortest predicted path to a transition with predicted reward > 0.5, never
  through one predicted to hurt (< -0.1). The agent follows the plan and RE-PLANS when the world refutes a prediction
  (the observed robot-relevant state differs from the predicted one), or when the plan runs out.
"""
import numpy as np
import torch

from village import GLOBAL_VOCAB

G = len(GLOBAL_VOCAB)


def reward_features(glob, glob_next):
    """[B, F]: per global field, one-hot of the change (-10..10) and of the new value (0..15), plus a bias."""
    g0, g1 = glob.astype(int), glob_next.astype(int)
    d = np.clip(g1 - g0, -10, 10) + 10
    out = [np.ones((len(g0), 1))]
    for f in range(G):
        out.append(np.eye(21)[d[:, f]])
        out.append(np.eye(16)[np.clip(g1[:, f], 0, 15)])
    return np.concatenate(out, 1)


class RewardModel:
    def fit(self, glob, glob_next, rew):
        X = reward_features(glob, glob_next)
        self.w, *_ = np.linalg.lstsq(X, rew.astype(np.float64), rcond=None)
        resid = rew - X @ self.w
        return float(np.abs(resid).max())

    def __call__(self, glob, glob_next):
        return reward_features(glob, glob_next) @ self.w


def robot_key(cells, glob):
    """Bytes keys of the robot-relevant predicted state, one per row."""
    pos = (cells[:, :, 3] == 1).argmax(1)
    facing = cells[np.arange(len(cells)), pos, 8]
    k = np.concatenate([pos[:, None], facing[:, None], glob[:, :3], cells[:, :, 0], cells[:, :, 1]], 1).astype(np.uint8)
    return [row.tobytes() for row in k]


class Planner:
    def __init__(self, model, reward, dev, depth=30, max_frontier=300, chunk=4096):
        self.model, self.reward, self.dev = model, reward, dev
        self.n_act = model.spec.n_actions
        self.depth, self.max_frontier, self.chunk = depth, max_frontier, chunk
        self.forwards = 0

    @torch.no_grad()
    def predict(self, cells, glob, acts):
        outc, outg = [], []
        for i in range(0, len(cells), self.chunk):
            c = torch.from_numpy(cells[i:i + self.chunk]).to(self.dev)
            g = torch.from_numpy(glob[i:i + self.chunk]).to(self.dev)
            a = torch.from_numpy(acts[i:i + self.chunk]).to(self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cl, gl, _ = self.model(c, g, a)
            outc.append(torch.stack([x.argmax(-1) for x in cl], -1).to(torch.uint8).cpu().numpy())
            outg.append(torch.stack([x.argmax(-1) for x in gl], -1).to(torch.uint8).cpu().numpy())
        self.forwards += len(cells)
        return np.concatenate(outc), np.concatenate(outg)

    def plan(self, cells, glob):
        """-> (actions, predicted robot keys after each action) of the shortest predicted path to reward, or None."""
        fc, fg = cells[None].copy(), glob[None].copy()
        paths = [[]]
        keys_path = [[]]
        seen = set(robot_key(fc, fg))
        for _ in range(self.depth):
            n = len(fc)
            acts = np.tile(np.arange(self.n_act, dtype=np.uint8), n)
            nc, ng = self.predict(np.repeat(fc, self.n_act, 0), np.repeat(fg, self.n_act, 0), acts)
            r = self.reward(np.repeat(fg, self.n_act, 0), ng)
            keys = robot_key(nc, ng)
            best = np.argmax(r)
            if r[best] > 0.5:
                i = best // self.n_act
                return paths[i] + [int(acts[best])], keys_path[i] + [keys[best]]
            keep = []
            for j in range(len(keys)):
                if r[j] > -0.1 and keys[j] not in seen:
                    seen.add(keys[j])
                    keep.append(j)
            if not keep:
                return None
            keep = keep[:self.max_frontier]
            fc, fg = nc[keep], ng[keep]
            paths = [paths[j // self.n_act] + [int(acts[j])] for j in keep]
            keys_path = [keys_path[j // self.n_act] + [keys[j]] for j in keep]
        return None


def run_episode(env, planner, rng):
    """Play one episode with plan-follow-replan. Returns (return, stats, plans made)."""
    cells, glob = env.obs()
    ret, plan, plans = 0.0, None, 0
    while not env.done:
        if not plan:
            out = planner.plan(cells, glob)
            plans += 1
            plan = list(zip(*out)) if out else [(int(rng.integers(0, 4)), None)]   # no plan found: explore
        a, expected = plan.pop(0)
        r, _ = env.step(a)
        ret += r
        cells, glob = env.obs()
        if expected is not None and robot_key(cells[None], glob[None])[0] != expected:
            plan = None                                              # refuted: re-plan from what actually happened
    return ret, env.summary(), plans


class RefutationLearner:
    """Learning while acting (E4b): when the world refutes a prediction, rehearse that transition until the model predicts
    it (its robot-relevant key matches, at most `max_updates`), each update = the refuted transition x `k` + `n_bg`
    rows of past experience (interleaving), gentle steps. Every transition seen while acting is kept, so later
    rehearsals can interleave the agent's own refutations too."""

    def __init__(self, model, data, dev, focus=2, lr=1e-4, k=8, n_bg=24, max_updates=30, seed=0,
                 slow=None, ema=0.98, pool=None, n_pool=16):
        """E4c options: `slow` = a copy the planner uses, updated as an EMA of `model`; `pool` = rows of past
        refutations to interleave (kind-balanced), n_pool of the n_bg background rows."""
        from e2_world import losses, robot_mask
        self.losses, self.robot_mask = losses, robot_mask
        self.model, self.data, self.dev, self.focus = model, data, dev, focus
        self.slow, self.ema, self.pool, self.n_pool = slow, ema, pool, n_pool
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.k, self.n_bg, self.max_updates = k, n_bg, max_updates
        self.rng = np.random.default_rng(seed)
        self.own = []                                   # this agent's refuted transitions
        self.updates = self.refutations = self.explained = 0

    def learn(self, cells, glob, a, ncells, nglob, planner):
        self.refutations += 1
        self.own.append((cells, glob, a, ncells, nglob))
        target = robot_key(ncells[None], nglob[None])[0]
        for _ in range(self.max_updates):
            bg = self.rng.integers(0, len(self.data["act"]), self.n_bg)
            if self.pool is not None:                   # kind-balanced past refutations
                bg[:self.n_pool] = self.pool[self.rng.integers(0, len(self.pool), self.n_pool)]
            own = [self.own[i] for i in self.rng.integers(0, len(self.own), max(0, self.k // 2))]
            rows = [(cells, glob, a, ncells, nglob)] * (self.k - len(own)) + own
            o = np.concatenate([np.stack([r[0] for r in rows]), self.data["obs"][bg]])
            g = np.concatenate([np.stack([r[1] for r in rows]), self.data["gob"][bg]])
            ac = np.concatenate([np.array([r[2] for r in rows], np.uint8), self.data["act"][bg]])
            nc = np.concatenate([np.stack([r[3] for r in rows]), self.data["nxt"][bg]])
            ng = np.concatenate([np.stack([r[4] for r in rows]), self.data["gnx"][bg]])
            T = lambda x: torch.from_numpy(x).to(self.dev)
            o, g, ac, nc, ng = T(o), T(g), T(ac), T(nc), T(ng)
            self.model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cl, gl, _ = self.model(o, g, ac)
            loss = self.losses(cl, gl, nc, ng, mask=self.robot_mask(o, self.focus))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self.model.eval()
            if self.slow is not None:
                with torch.no_grad():
                    for ps, pf in zip(self.slow.parameters(), self.model.parameters()):
                        ps.mul_(self.ema).add_(pf.detach(), alpha=1 - self.ema)
            self.updates += 1
            pc, pg = planner.predict(cells[None], glob[None], np.array([a], np.uint8))
            if robot_key(pc, pg)[0] == target:
                self.explained += 1
                return True
        return False


def run_episode_learning(env, planner, learner, rng):
    """plan-follow-replan, and learn from every refutation before re-planning."""
    cells, glob = env.obs()
    ret, plan, plans = 0.0, None, 0
    while not env.done:
        if not plan:
            out = planner.plan(cells, glob)
            plans += 1
            plan = list(zip(*out)) if out else [(int(rng.integers(0, 4)), None) for _ in range(16)]   # no plan: explore
        a, expected = plan.pop(0)
        if expected is None and getattr(learner, "predict_always", False):   # E5b: every action is a prediction
            pc, pg = planner.predict(cells[None], glob[None], np.array([a], np.uint8))
            expected = robot_key(pc, pg)[0]
            exploring = True
        else:
            exploring = False
        r, _ = env.step(a)
        ret += r
        nc, ng = env.obs()
        if expected is not None and robot_key(nc[None], ng[None])[0] != expected:
            learner.learn(cells, glob, a, nc, ng, planner)       # refuted: work out why, then re-plan
            plan = None
        cells, glob = nc, ng
    return ret, env.summary(), plans


def sleep_on_refutations(learner, planner, n_updates=100, n_own=8, n_pool=16, n_stream=8):
    """E4d: after an episode, consolidate the agent's own refutations: group them by surprise kind (with the current
    model), rehearse evenly across kinds, retire a refutation once its surprise has halved, interleave the kind-balanced
    pool and logged rows; gentle steps, the slow copy follows as an EMA."""
    from e2_world import losses, robot_mask, robot_pos
    if not learner.own:
        return 0
    dev, rng = learner.dev, learner.rng
    T = lambda x: torch.from_numpy(np.asarray(x)).to(dev)

    def own_batch(idx):
        rows = [learner.own[i] for i in idx]
        return (np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]), np.array([r[2] for r in rows], np.uint8),
                np.stack([r[3] for r in rows]), np.stack([r[4] for r in rows]))

    with torch.no_grad():                                         # kinds and surprise of the own refutations
        o, g, a, nc, ng = (T(x) for x in own_batch(range(len(learner.own))))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cl, gl, _ = learner.model(o, g, a)
        _, s0, kinds = losses(cl, gl, nc, ng, per_row="sig", mask=robot_mask(o, learner.focus), robot_pos=robot_pos(o))
        s0 = s0.cpu().numpy()
    open_ = {}
    for i, k in enumerate(kinds):
        open_.setdefault(int(k), []).append(i)
    done = 0
    for _ in range(n_updates):
        live = [k for k in open_ if open_[k]]
        if not live:
            break
        pick = [open_[k][rng.integers(len(open_[k]))] for k in rng.choice(live, n_own)]
        oo, og, oa, onc, ong = own_batch(pick)
        bg = np.concatenate([learner.pool[rng.integers(0, len(learner.pool), n_pool)],
                             rng.integers(0, len(learner.data["act"]), n_stream)])
        d = learner.data
        o = T(np.concatenate([oo, d["obs"][bg]]))
        g = T(np.concatenate([og, d["gob"][bg]]))
        a = T(np.concatenate([oa, d["act"][bg]]))
        nc = T(np.concatenate([onc, d["nxt"][bg]]))
        ng = T(np.concatenate([ong, d["gnx"][bg]]))
        learner.model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cl, gl, _ = learner.model(o, g, a)
        loss, tot = losses(cl, gl, nc, ng, per_row=True, mask=robot_mask(o, learner.focus))
        learner.opt.zero_grad(set_to_none=True)
        loss.backward()
        learner.opt.step()
        learner.model.eval()
        with torch.no_grad():
            for ps, pf in zip(learner.slow.parameters(), learner.model.parameters()):
                ps.mul_(learner.ema).add_(pf.detach(), alpha=1 - learner.ema)
        st = tot[:n_own].cpu().numpy()
        for i, sv in zip(pick, st):                               # retire when explained
            for k in live:
                if i in open_[k] and sv < 0.5 * s0[i]:
                    open_[k].remove(i)
                    done += 1
                    break
        learner.updates += 1
    return done
