"""Oracle behavior-cloning dataset.

Each procedural layout is rolled out by the BFS oracle into an optimal action
sequence; every decision point becomes a training pair:

    (window of W recent frames, the actions between them)  ->  the oracle's action

Frames are stored compactly (uint8, one copy per episode) and windows are built on
the fly in __getitem__, so the memory cost is ~episodes, not ~decision-points.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from arc_agi_3 import Environment
from arc_agi_3.games import LockPath
from arc_agi_3.oracle import solve_level

# An episode: (frames uint8 (L+1, 64, 64), action ids [a_0 .. a_{L-1}]).
Episode = Tuple[torch.Tensor, List[int]]


def _grid_u8(grid) -> torch.Tensor:
    return torch.tensor(grid, dtype=torch.uint8)


def rollout_oracle(layout: List[str]) -> Optional[Episode]:
    """Solve `layout` with the oracle and record (frames, action ids)."""
    env = Environment(LockPath([layout]))
    frame = env.reset()
    path = solve_level(env.game)
    if not path:                       # unsolvable or trivially solved
        return None
    frames = [_grid_u8(frame.grid)]
    actions: List[int] = []
    for action in path:
        frame = env.step(action)
        frames.append(_grid_u8(frame.grid))
        actions.append(action.value)
    return torch.stack(frames), actions


class OracleBCDataset(Dataset):
    """(frame-window, inter-frame actions) -> oracle action, over many layouts."""

    def __init__(self, layouts: List[List[str]], window: int = 2):
        self.window = window
        self.episodes: List[Episode] = []
        self.index: List[Tuple[int, int]] = []
        for layout in layouts:
            episode = rollout_oracle(layout)
            if episode is None:
                continue
            ei = len(self.episodes)
            self.episodes.append(episode)
            for t in range(len(episode[1])):
                self.index.append((ei, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ei, t = self.index[i]
        frames, actions = self.episodes[ei]
        w = self.window
        # Window frame indices end at t (the state the agent decides from), clamped.
        idxs = [max(0, t - w + 1 + k) for k in range(w)]
        fw = frames[idxs].long()                              # (W, 64, 64)
        # Inter-frame action ids: the action on each in-window transition, -1 if padded.
        aw = []
        for k in range(w - 1):
            j0, j1 = idxs[k], idxs[k + 1]
            aw.append(actions[j0] if j1 == j0 + 1 else -1)
        aw_t = torch.tensor(aw, dtype=torch.long)             # (W-1,)
        target = torch.tensor(actions[t], dtype=torch.long)   # scalar action id
        return fw, aw_t, target


def build_dataset(layouts: List[List[str]], window: int = 2) -> OracleBCDataset:
    return OracleBCDataset(layouts, window=window)


def materialize(ds: OracleBCDataset):
    """Stack every decision point into dense tensors for GPU-resident training.

    Returns (frames uint8 (N, W, 64, 64), actions long (N, W-1), targets long (N,)).
    The whole BC dataset is small (~tens of MB), so staging it on-device once and
    batching by random index avoids the DataLoader/CPU path that starves a tiny
    model's GPU.
    """
    n, w = len(ds), ds.window
    frames = torch.empty((n, w, 64, 64), dtype=torch.uint8)
    actions = torch.empty((n, w - 1), dtype=torch.long)
    targets = torch.empty((n,), dtype=torch.long)
    for i in range(n):
        fw, aw, tg = ds[i]
        frames[i] = fw.to(torch.uint8)
        actions[i] = aw
        targets[i] = tg
    return frames, actions, targets
