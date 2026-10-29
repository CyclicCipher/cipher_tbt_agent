    def displayed(self, col, static):
        """Part 31: where the screen shows this column's shape again, as a delineated object: up to a quarter-turn,
        under one colour map that keeps the column's distinct colours distinct, outside the column itself. Returns the
        states that would put the column on each copy. The transform from the column to a copy (the shift, the turn,
        the colour map) is exactly how they differ; reaching the copy's pose is the goal hypothesis it suggests."""
        f = col.frame
        if f.kind != "translation":
            return []
        colours = np.asarray(f.colours, int)
        ground = self.v1.ground
        own = set(map(tuple, f.place(col.state).tolist()))
        out = []
        for r in range(4):
            p = f.place((0, 0, r))
            rel = p - p[0]
            x0, x1 = -rel[:, 0].min(), W - rel[:, 0].max()
            y0, y1 = -rel[:, 1].min(), H - rel[:, 1].max()
            if x1 <= x0 or y1 <= y0:
                continue
            X, Y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            pos = np.stack([X.ravel(), Y.ravel()], 1)
            vals = static[pos[:, None, 1] + rel[None, :, 1], pos[:, None, 0] + rel[None, :, 0]]
            ok = (vals != ground).any(1)
            firsts = {}
            for c in set(colours.tolist()):                 # one colour map: each colour of the column maps to one
                idx = np.nonzero(colours == c)[0]           # colour, and distinct colours to distinct colours
                ok &= (vals[:, idx] == vals[:, idx[:1]]).all(1)
                firsts[c] = vals[:, idx[0]]
            cs = sorted(firsts)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    ok &= firsts[cs[i]] != firsts[cs[j]]
            shape = set(map(tuple, rel.tolist()))
            ring = [(x + dx, y + dy) for x, y in shape for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if (x + dx, y + dy) not in shape]
            for k in np.nonzero(ok)[0]:                     # delineated: no colour runs on across its edge
                px, py = pos[k]
                cells = {(px + x, py + y) for x, y in shape}
                if cells & own:
                    continue
                edge_ok = True
                for x, y in ring:
                    ex, ey = px + x, py + y
                    if 0 <= ex < W and 0 <= ey < H:
                        inside = [(px + x - dx, py + y - dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                                  if (x - dx, y - dy) in shape]
                        if any(static[ey, ex] == static[iy, ix] for ix, iy in inside):
                            edge_ok = False
                            break
                if edge_ok:
                    out.append((int(px - p[0][0]), int(py - p[0][1]), r))
            if len(out) > MATCH_MAX:                        # too many copies: the shape says nothing about a goal
                return []
        return out

