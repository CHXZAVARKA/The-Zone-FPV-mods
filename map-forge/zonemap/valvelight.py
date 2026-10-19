"""Valve's own light - the lightmaps vrad baked into a Source map - queryable
at any point: the light a surface there was given, as linear light in the
lightmap's units (Source shows a surface at 128 with its albedo unchanged:
255 with the engine's overbright of 2).

tools/bake_glb.py reads it to set the Cycles bake against the original; the
forge lights the 3D skybox room with it (extras.py), since that room is drawn
by the map_extras mod and never baked.
"""
import collections
import numpy as np

from .lightmap import _decode

GRID_UNITS = 128.0
DISP_CELL = 512.0        # displacement lookup grid, Hammer units
DISP_PAD = 256.0         # how far a displacement may rise off its base quad


class LightField:
    """Valve's lightmap, queryable at any world point."""

    def __init__(self, bsp, verbose=True, keep=None):
        """keep: an optional test on a face's centre (Hammer units) - only the
        faces it passes are read, which is far quicker on a big map when only
        one part of it is wanted"""
        self.bsp = bsp
        raw = bsp.lump(8)
        if not len(raw):
            raw = bsp.lump(53)
        self.raw = raw
        self.by_plane = collections.defaultdict(list)
        self.grid = {}
        self.by_cell = collections.defaultdict(list)
        self._cellcache = {}
        self._nearcache = {}
        f = bsp.faces
        acc = collections.defaultdict(lambda: [np.zeros(3), 0])
        self.faces = []
        for fi in range(len(f)):
            if int(f[fi]['lightofs']) < 0:
                continue
            loop = bsp.face_loop(fi)
            if len(loop) < 3:
                continue
            q = bsp.verts[loop].astype(np.float64)
            if keep is not None and not keep(q.mean(0)):
                continue
            lm = self._luxels(fi)
            if lm is None:
                continue
            pn = np.array(bsp.planes[int(f[fi]['planenum'])]['normal'], np.float64)
            d = float(pn @ q[0])
            ti = bsp.texinfo[int(f[fi]['texinfo'])]
            rec = (pn, d, np.asarray(ti['lm_vecs'], np.float64),
                   np.asarray(f[fi]['lm_mins'], np.float64), lm, q.mean(0),
                   q.min(0), q.max(0))
            idx = len(self.faces)
            self.faces.append(rec)
            ax = int(np.argmax(np.abs(pn)))
            self.by_plane[(ax, 1 if pn[ax] >= 0 else -1,
                           int(round(d / 16.0)))].append(idx)
            if int(f[fi]['dispinfo']) >= 0:
                # A displacement's geometry is nowhere near its face plane, so
                # a plane lookup never finds it - and de_aztec is mostly
                # terrain. Its lightmap axes are the texture axes, which for a
                # ground quad are horizontal, so the DISPLACED point still maps
                # to the right luxel; only the plane test has to go. Index it
                # by the cells its base quad covers instead.
                lo = np.floor((q.min(0) - DISP_PAD) / DISP_CELL).astype(int)
                hi = np.floor((q.max(0) + DISP_PAD) / DISP_CELL).astype(int)
                for cx in range(lo[0], hi[0] + 1):
                    for cy in range(lo[1], hi[1] + 1):
                        for cz in range(lo[2], hi[2] + 1):
                            self.by_cell[(cx, cy, cz)].append(idx)
            c = rec[5]
            k = tuple(np.floor(c / GRID_UNITS).astype(int))
            a = acc[k]
            a[0] += lm.reshape(-1, 3).mean(0)
            a[1] += 1
        self.grid = {k: v[0] / max(v[1], 1) for k, v in acc.items()}
        self.gridkeys = np.array(list(self.grid.keys()), np.int64) if self.grid \
            else np.zeros((0, 3), np.int64)
        self.gridvals = np.array(list(self.grid.values()), np.float64) if self.grid \
            else np.zeros((0, 3), np.float64)
        self.median = (np.median(self.gridvals, axis=0) if len(self.gridvals)
                       else np.ones(3))
        if verbose:
            print(f'      light: {len(self.faces)} lit faces, '
                  f'{len(self.grid)} grid cells')

    def _luxels(self, fi):
        f = self.bsp.faces[fi]
        off = int(f['lightofs'])
        w = int(f['lm_size'][0]) + 1
        h = int(f['lm_size'][1]) + 1
        if w < 1 or h < 1 or off + w * h * 4 > len(self.raw):
            return None
        try:
            return _decode(self.raw[off:off + w * h * 4], w * h).reshape(h, w, 3)
        except ValueError:
            return None

    @staticmethod
    def _lux(rec, pts):
        lv, mins = rec[2], rec[3]
        s = pts @ lv[0, :3] + lv[0, 3] - mins[0]
        t = pts @ lv[1, :3] + lv[1, 3] - mins[1]
        return s, t

    @staticmethod
    def _bilinear(lm, s, t):
        h, w = lm.shape[:2]
        s = np.clip(s, 0, w - 1); t = np.clip(t, 0, h - 1)
        x0 = np.floor(s).astype(np.int64); y0 = np.floor(t).astype(np.int64)
        x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
        fx = (s - x0)[:, None]; fy = (t - y0)[:, None]
        a = lm[y0, x0] * (1 - fx) + lm[y0, x1] * fx
        b = lm[y1, x0] * (1 - fx) + lm[y1, x1] * fx
        return a * (1 - fy) + b * fy

    def sample(self, pts, n, stats=None):
        """(N,3) Hammer-space points on a plane with normal `n` -> (N,3) light."""
        out = np.zeros((len(pts), 3), np.float64)
        todo = np.ones(len(pts), bool)
        n = np.asarray(n, np.float64)
        ax = int(np.argmax(np.abs(n)))
        sg = 1 if n[ax] >= 0 else -1
        d = pts @ n
        dk = int(round(float(np.median(d)) / 16.0))
        cand = []
        for k in (dk - 1, dk, dk + 1):
            cand += self.by_plane.get((ax, sg, k), [])
        for fi in cand:
            if not todo.any():
                break
            rec = self.faces[fi]
            if rec[0] @ n < 0.99:
                continue
            near = todo & (np.abs(d - rec[1]) < 2.0)
            if not near.any():
                continue
            s, t = self._lux(rec, pts[near])
            h, w = rec[4].shape[:2]
            inside = (s >= -0.75) & (t >= -0.75) & (s <= w - 0.25) & (t <= h - 0.25)
            if not inside.any():
                continue
            sel = np.nonzero(near)[0][inside]
            out[sel] = self._bilinear(rec[4], s[inside], t[inside])
            todo[sel] = False
        if todo.any():
            self._disp_pass(pts, out, todo)
        if stats is not None:
            stats['exact'] += int((~todo).sum())
        if todo.any():
            out[todo] = self._fallback(pts[todo], n, ax, sg, stats)
            if stats is not None:
                stats['inexact'] += int(todo.sum())
        return out

    def _disp_pass(self, pts, out, todo):
        seen = set()
        key = np.floor(pts / DISP_CELL).astype(np.int64)
        for k in map(tuple, np.unique(key, axis=0)):
            for fi in self.by_cell.get(k, ()):
                if fi in seen or not todo.any():
                    continue
                seen.add(fi)
                rec = self.faces[fi]
                near = todo & (pts >= rec[6] - DISP_PAD).all(1) \
                    & (pts <= rec[7] + DISP_PAD).all(1)
                if not near.any():
                    continue
                s, t = self._lux(rec, pts[near])
                h, w = rec[4].shape[:2]
                inside = (s >= -0.75) & (t >= -0.75) & (s <= w - 0.25) & (t <= h - 0.25)
                if not inside.any():
                    continue
                sel = np.nonzero(near)[0][inside]
                out[sel] = self._bilinear(rec[4], s[inside], t[inside])
                todo[sel] = False

    def _fallback(self, pts, n, ax, sg, stats=None):
        """Nearest same-facing lit surface, else the grid."""
        out = np.zeros((len(pts), 3), np.float64)
        d = pts @ n
        dk = int(round(float(np.median(d)) / 16.0))
        cen = pts.mean(0)
        # memoised: this scan is per-texel-group and was the slowest thing in
        # the bake, repeated for every chart standing on the same shell
        ck = (ax, sg, dk, int(cen[0] // 256), int(cen[1] // 256), int(cen[2] // 256))
        hit = self._nearcache.get(ck, 0)
        if hit != 0:
            best, bestd = hit
        else:
            best = None
            bestd = 1e18
            for k in range(dk - 8, dk + 9):          # +-128 units
                for fi in self.by_plane.get((ax, sg, k), []):
                    rec = self.faces[fi]
                    if rec[0] @ n < 0.9:
                        continue
                    dist = float(np.linalg.norm(rec[5] - cen))
                    if dist < bestd:
                        bestd, best = dist, rec
            self._nearcache[ck] = (best, bestd)
        if best is not None and bestd < 512.0:
            s, t = self._lux(best, pts)
            h, w = best[4].shape[:2]
            out[:] = self._bilinear(best[4], np.clip(s, 0, w - 1),
                                    np.clip(t, 0, h - 1))
            if stats is not None:
                stats['near'] += len(pts)
            return out
        if self.grid:
            # one lookup per distinct cell, not per texel: a chart's texels
            # nearly all land in the same cell, and scanning every cell per
            # point was the slowest thing in the bake
            key = np.floor(pts / GRID_UNITS).astype(np.int64)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            vals = np.empty((len(uniq), 3))
            for i, k in enumerate(map(tuple, uniq)):
                vals[i] = self._cell(k)
            out[:] = vals[inv]
            if stats is not None:
                stats['grid'] += len(pts)
        else:
            out[:] = self.median
        return out

    def _cell(self, k, max_r=6):
        hit = self._cellcache.get(k)
        if hit is not None:
            return hit
        for r in range(max_r + 1):
            best, bd = None, 1e18
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if r and max(abs(dx), abs(dy), abs(dz)) != r:
                            continue
                        v = self.grid.get((k[0] + dx, k[1] + dy, k[2] + dz))
                        if v is not None:
                            d = dx * dx + dy * dy + dz * dz
                            if d < bd:
                                bd, best = d, v
            if best is not None:
                self._cellcache[k] = best
                return best
        self._cellcache[k] = self.median
        return self.median
