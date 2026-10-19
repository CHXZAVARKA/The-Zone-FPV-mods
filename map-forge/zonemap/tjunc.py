"""Close the cracks where one surface's corner sits on another's edge.

A T-junction: face A's edge runs straight past a corner of face B (or of two
faces of B's that meet there). On screen A's edge and B's two edges are
rasterised separately, and float rounding leaves pixel-wide gaps along the
seam that show whatever is behind - a dotted line of sky or void along a
wall's foot or a ceiling's edge. vbsp mends its own faces this way (their
edge lists carry every corner that touches them), but the faces this
converter makes - rebuilt hidden faces, displacements, cut blends, brush
entities, the sky surfaces - were not mended against them or each other:
gm_br_pitfalls' spawn room alone had 179 corners lying on others' edges.

So, over all the world's triangles together: every corner lying on another
triangle's edge (within EPS of it, not at its ends) is put into that edge,
and the triangle is cut through it - its sharpest corner clipped off again
and again, so no corner is added and no piece is flat - with its normals and
UVs carried over. The new corners are the very same positions as the ones they
meet, so the seam then closes exactly.
"""
import collections
import numpy as np

EPS = 1e-4          # m off the edge's line to count as on it
END = 1e-3          # m from an edge's end: that is the end, not a T
Q = 1e-5            # m: positions this close are one


def _tris(chunks):
    """every triangle of every chunk -> (P (T,3,3) f64, chunk id, tri id)"""
    P, cid, tid = [], [], []
    for ci, c in enumerate(chunks):
        pos, idx = np.asarray(c[2], np.float64), np.asarray(c[5]).reshape(-1, 3)
        if not len(idx):
            continue
        P.append(pos[idx]); cid.append(np.full(len(idx), ci)); tid.append(np.arange(len(idx)))
    if not P:
        return np.zeros((0, 3, 3)), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(P), np.concatenate(cid), np.concatenate(tid)


def find(P):
    """{(triangle, edge k): [(t along the edge, point)]} - the corners lying
    on each triangle's edges"""
    V = P.reshape(-1, 3)
    key = np.round(V / Q).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    U = V[first]                                  # one position per corner
    vid = inv.reshape(-1, 3)
    ea = np.concatenate([vid[:, 0], vid[:, 1], vid[:, 2]])
    eb = np.concatenate([vid[:, 1], vid[:, 2], vid[:, 0]])
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    ekey = lo * len(U) + hi
    uek, eidx = np.unique(ekey, return_inverse=True)
    E0, E1 = uek // len(U), uek % len(U)
    A, Bv = U[E0], U[E1]
    D = Bv - A
    L = np.linalg.norm(D, axis=1)
    hits = collections.defaultdict(list)           # unique edge -> [(t, vertex)]
    # axis-aligned edges (most of a brush map): the corners with the same two
    # other coordinates, between the ends along the third
    done = np.zeros(len(uek), bool)
    for ax in range(3):
        o1, o2 = [k for k in range(3) if k != ax]
        on = (np.abs(D[:, o1]) < EPS) & (np.abs(D[:, o2]) < EPS) & (L > 2 * END)
        if not on.any():
            continue
        g = collections.defaultdict(list)
        q1 = np.round(U[:, o1] / (2 * EPS)).astype(np.int64)
        q2 = np.round(U[:, o2] / (2 * EPS)).astype(np.int64)
        order = np.lexsort((U[:, ax], q2, q1))
        k1, k2 = q1[order], q2[order]
        cut = np.nonzero((np.diff(k1) != 0) | (np.diff(k2) != 0))[0] + 1
        starts = np.concatenate([[0], cut]); ends = np.concatenate([cut, [len(order)]])
        groups = {(int(k1[s]), int(k2[s])): order[s:e] for s, e in zip(starts, ends)}
        for e in np.nonzero(on)[0]:
            a = A[e]
            gk = (int(round(a[o1] / (2 * EPS))), int(round(a[o2] / (2 * EPS))))
            members = []
            for d1 in (-1, 0, 1):
                for d2 in (-1, 0, 1):
                    m = groups.get((gk[0] + d1, gk[1] + d2))
                    if m is not None:
                        members.append(m)
            if not members:
                continue
            m = np.concatenate(members)
            x0, x1 = sorted((A[e, ax], Bv[e, ax]))
            c = U[m, ax]
            sel = m[(c > x0 + END) & (c < x1 - END)
                    & (np.abs(U[m, o1] - a[o1]) < EPS) & (np.abs(U[m, o2] - a[o2]) < EPS)]
            if len(sel):
                t = (U[sel, ax] - A[e, ax]) / D[e, ax]
                hits[e] = list(zip(t.tolist(), sel.tolist()))
        done |= on
    # the rest: a grid of corners, walked along each edge
    rest = np.nonzero(~done & (L > 2 * END))[0]
    if len(rest):
        h = 0.5
        cells = collections.defaultdict(list)
        for i, ck in enumerate(map(tuple, np.floor(U / h).astype(np.int64))):
            cells[ck].append(i)
        for e in rest:
            n = int(L[e] / (h * 0.5)) + 2
            s = A[e] + np.outer(np.linspace(0.0, 1.0, n), D[e])
            ks = {tuple(k) for k in np.floor(s / h).astype(np.int64)}
            cand = [i for k in ks for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                    for i in cells.get((k[0] + dx, k[1] + dy, k[2] + dz), ())]
            if not cand:
                continue
            cand = np.unique(np.array(cand))
            p = U[cand]
            t = ((p - A[e]) @ D[e]) / (L[e] * L[e])
            perp = np.linalg.norm(A[e] + np.outer(t, D[e]) - p, axis=1)
            ok = (t * L[e] > END) & ((1 - t) * L[e] > END) & (perp < EPS)
            if ok.any():
                hits[e] = list(zip(t[ok].tolist(), cand[ok].tolist()))
    # back to triangle edges, t measured from the triangle's own edge start
    out = collections.defaultdict(list)
    T = len(P)
    tri_e = eidx.reshape(3, T).T                   # (T, 3) unique edge of each triangle edge
    starts = vid                                    # edge k runs from corner k to k+1
    if not hits:
        return out, U
    hit_edges = np.zeros(len(uek), bool)
    hit_edges[list(hits.keys())] = True
    rows, cols = np.nonzero(hit_edges[tri_e])
    for ti, k in zip(rows.tolist(), cols.tolist()):
        e = tri_e[ti, k]
        flip = starts[ti, k] != E0[e]
        own = set(vid[ti].tolist())
        # a sliver's own third corner can lie that close to its long edge:
        # that is not another face's corner
        lst = [((1.0 - t) if flip else t, v) for t, v in hits[e] if v not in own]
        if lst:
            out[(ti, k)] = sorted(lst)
    return out, U


def _ears(poly, pts):
    """a convex polygon whose sides carry extra corners in line with them ->
    triangles, none of them flat, no corner added: clip the sharpest corner
    off, again and again"""
    ring = list(poly)
    out = []
    # the triangle's own facing: a corner is an ear only if it turns that way
    # (a corner put in a hair off its edge can turn the other way)
    q = pts[ring]
    fn = np.zeros(3)
    for i in range(len(q)):
        fn += np.cross(q[i - 1], q[i])
    fn /= max(float(np.linalg.norm(fn)), 1e-15)
    while len(ring) > 3:
        best, bi = -1.0, -1
        n = len(ring)
        for i in range(n):
            a, b, c = pts[ring[i - 1]], pts[ring[i]], pts[ring[(i + 1) % n]]
            u, v = b - a, c - b
            lu, lv = np.linalg.norm(u), np.linalg.norm(v)
            if lu < 1e-12 or lv < 1e-12:
                continue
            s = float(np.cross(u, v) @ fn) / (lu * lv)   # sin of the turn, signed
            if s > best:
                best, bi = s, i
        if bi < 0 or best < 1e-9:
            break
        n = len(ring)
        out.append((ring[bi - 1], ring[bi], ring[(bi + 1) % n]))
        ring.pop(bi)
    if len(ring) == 3:
        out.append(tuple(ring))
    elif len(ring) > 3:
        # nothing left that turns (a sliver gone flat): fan what remains rather
        # than leave a hole
        out += [(ring[0], a, b) for a, b in zip(ring[1:-1], ring[2:])]
    return out


def fix(chunks, verbose=True):
    """chunks: [(material, cell, pos, nrm, uv, idx)] -> (chunks, stats)"""
    P, cid, tid = _tris(chunks)
    st = collections.Counter()
    if not len(P):
        return chunks, st
    hits, U = find(P)
    if not hits:
        return chunks, st
    by_chunk = collections.defaultdict(dict)       # chunk -> {tri: {edge: [(t, v)]}}
    for (ti, k), lst in hits.items():
        by_chunk[int(cid[ti])].setdefault(int(tid[ti]), {})[k] = lst
        st['corners_put_in'] += len(lst)
    new = list(chunks)
    for ci, tris in by_chunk.items():
        m, cell, pos, nrm, uv, idx = chunks[ci]
        pos = np.asarray(pos, np.float32); nrm = np.asarray(nrm, np.float32); uv = np.asarray(uv, np.float32)
        I = np.asarray(idx).reshape(-1, 3).astype(np.int64)
        P2, N2, U2 = [pos], [nrm], [uv]
        nv = len(pos)
        keep = np.ones(len(I), bool)
        add = []

        def vert(p, w, t3):
            # a new corner: at p, normal and uv by barycentric weight w
            nonlocal nv
            P2.append(np.asarray(p, np.float32)[None, :])
            n = (w[0] * nrm[t3[0]] + w[1] * nrm[t3[1]] + w[2] * nrm[t3[2]])
            n = n / max(float(np.linalg.norm(n)), 1e-9)
            N2.append(np.asarray(n, np.float32)[None, :])
            U2.append(np.asarray(w[0] * uv[t3[0]] + w[1] * uv[t3[1]] + w[2] * uv[t3[2]], np.float32)[None, :])
            nv += 1
            return nv - 1
        for ti, edges in tris.items():
            t3 = I[ti]
            keep[ti] = False
            poly = []                               # (vertex index) round the triangle
            for k in range(3):
                poly.append(int(t3[k]))
                for t, v in edges.get(k, ()):
                    w = np.zeros(3); w[k] = 1.0 - t; w[(k + 1) % 3] = t
                    poly.append(vert(U[v], w, t3))
            pts = np.concatenate(P2).astype(np.float64) if len(P2) > 1 else pos.astype(np.float64)
            add += _ears(poly, pts)
            st['triangles_cut'] += 1
        I2 = np.concatenate([I[keep], np.array(add, np.int64).reshape(-1, 3)])
        st['triangles_added'] += len(add) - len(tris)
        shape = (-1, 3) if np.asarray(idx).ndim == 2 else (-1,)
        new[ci] = (m, cell, np.concatenate(P2), np.concatenate(N2), np.concatenate(U2),
                   I2.astype(np.uint32).reshape(shape))
    if verbose:
        print(f"[2/5] T-junctions: {st['corners_put_in']:,} corners lying on other faces' edges put "
              f"into them ({st['triangles_cut']:,} triangles cut, {st['triangles_added']:,} added)")
    return new, st
