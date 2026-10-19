#!/usr/bin/env python3
"""Put a baked map's light onto pages by location, without baking again.

bake_glb --zone-v3 now packs its light pages by where the charts are
(zonemap/pagelayout.py), so each page - each node the game draws - is one
compact part of the map and Godot can cull it, by view and by the map's
occluders. A map baked before that has pages that each span the whole map,
so nothing is ever culled. This rearranges such a bake as if it had been made
that way: it finds the charts on the old pages from the triangles' light UVs,
groups them by location, copies each chart's light texel for texel onto new
pages, moves the UVs by the same whole texels, and writes the occluders. The
light itself is not worked out again, so the map looks exactly as it did.

    python tools/relayout_v3.py out/<map>/<map>_v3_zone_v3 [--source out/<map>/<map>.glb]

The bake folder gets the new map.glb, lightmaps/ and bake_info.json, and an
occluder.bin; the old ones are kept beside them as *_before_relayout. Then
pack it as the bake would have (tools/pack_v3.py, or the Godot command
bake_glb prints); the textures are reused from the installed map, so only the
light is packed again.

--source is the forge's .glb the bake was made from, for the occluders (its
walls and floors, not its props); without it they come from the light pages
themselves, props and all.

Each step keeps its results in --work (default: beside the bake folder), so a
run stopped part way picks up where it was: --step scan, charts, light, glb,
finish run one at a time.
"""
import argparse, json, math, os, re, shutil, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from zonemap import pagelayout as PL, occluder as OCC

PAGE_RE = re.compile(r'^lm(\d+)(?:s\d+)?_zlms-(\d+)$')
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
CT = {5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}


# ---- the .glb, read in place --------------------------------------------------
class Glb:
    def __init__(self, path):
        with open(path, 'rb') as f:
            assert f.read(4) == b'glTF', f'not a .glb: {path}'
            f.read(8)
            ln, _ = struct.unpack('<II', f.read(8))
            self.j = json.loads(f.read(ln))
            self.bin_at = 12 + 8 + ln + 8
            f.seek(12 + 8 + ln)
            self.bln = struct.unpack('<II', f.read(8))[0]
        self.mm = np.memmap(path, np.uint8, 'r', offset=self.bin_at, shape=(self.bln,))

    def acc(self, i, rows=None):
        a = self.j['accessors'][i]
        v = self.j['bufferViews'][a['bufferView']]
        n = NC[a['type']]
        dt = np.dtype(CT[a['componentType']])
        off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        stride = v.get('byteStride') or n * dt.itemsize
        arr = np.ndarray((a['count'], n), dt, buffer=self.mm, offset=off, strides=(stride, dt.itemsize))
        out = arr if rows is None else arr[rows]
        out = np.array(out)
        return out.reshape(-1) if n == 1 else out

    def view(self, i):
        v = self.j['bufferViews'][i]
        o = v.get('byteOffset', 0)
        return self.mm[o:o + v['byteLength']]


# ---- EXR (the uncompressed scanline files bake_glb writes) --------------------
def read_exr(path):
    d = np.fromfile(path, np.uint8)
    p = 8
    chans, w, h = [], None, None
    while d[p] != 0:
        e = p
        while d[e]:
            e += 1
        name = bytes(d[p:e]).decode(); p = e + 1
        e = p
        while d[e]:
            e += 1
        typ = bytes(d[p:e]).decode(); p = e + 1
        size = int(np.frombuffer(d[p:p + 4].tobytes(), '<i4')[0]); p += 4
        val = d[p:p + size].tobytes(); p += size
        if name == 'channels':
            q = 0
            while val[q] != 0:
                e = val.index(b'\0', q)
                chans.append((val[q:e].decode(), struct.unpack('<i', val[e + 1:e + 5])[0]))
                q = e + 1 + 16
        elif name == 'dataWindow':
            x0, y0, x1, y1 = struct.unpack('<iiii', val)
            w, h = x1 - x0 + 1, y1 - y0 + 1
        elif name == 'compression' and val != b'\0':
            raise SystemExit(f'{path}: compressed EXR, not one bake_glb wrote')
    p += 1 + 8 * h
    dt = {1: '<f2', 2: '<f4'}[chans[0][1]]
    sz = np.dtype(dt).itemsize
    lines = d[p:p + h * (8 + w * len(chans) * sz)].reshape(h, -1)[:, 8:]
    planes = lines.copy().view(dt).reshape(h, len(chans), w)
    out = {c: planes[:, k, :] for k, (c, _) in enumerate(chans)}
    return np.stack([out['R'], out['G'], out['B']], -1).astype(np.float32)


def write_exr_half(path, rgb):
    """bake_glb.write_exr, with half floats (the packer turns the light into
    half floats anyway): half the size"""
    h, w = rgb.shape[:2]

    def attr(name, typ, data):
        return name.encode() + b'\0' + typ.encode() + b'\0' + struct.pack('<i', len(data)) + data
    ch = b''.join(c.encode() + b'\0' + struct.pack('<iBBBBii', 1, 0, 0, 0, 0, 1, 1) for c in 'BGR') + b'\0'
    box = struct.pack('<iiii', 0, 0, w - 1, h - 1)
    hdr = (struct.pack('<ii', 20000630, 2) + attr('channels', 'chlist', ch) +
           attr('compression', 'compression', b'\0') + attr('dataWindow', 'box2i', box) +
           attr('displayWindow', 'box2i', box) + attr('lineOrder', 'lineOrder', b'\0') +
           attr('pixelAspectRatio', 'float', struct.pack('<f', 1.0)) +
           attr('screenWindowCenter', 'v2f', struct.pack('<ff', 0, 0)) +
           attr('screenWindowWidth', 'float', struct.pack('<f', 1.0)) + b'\0')
    row = w * 3 * 2
    start = len(hdr) + 8 * h
    offs = np.arange(h, dtype=np.uint64) * (8 + row) + start
    planar = np.ascontiguousarray(rgb[:, :, ::-1].transpose(0, 2, 1).astype('<f2'))
    lines = np.empty((h, 8 + row), np.uint8)
    lines[:, :4] = np.arange(h, dtype='<i4').view(np.uint8).reshape(h, 4)
    lines[:, 4:8] = np.full(h, row, '<i4').view(np.uint8).reshape(h, 4)
    lines[:, 8:] = planar.reshape(h, -1).view(np.uint8)
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(offs.astype('<u8').tobytes())
        f.write(lines.tobytes())


def fill_empty(img, mask):
    """bake_glb.fill_empty: every texel outside mask takes its baked
    neighbours' light (pull-push), so halving and mips never average in black"""
    if mask.all() or not mask.any():
        return img
    levels = [(img * mask[..., None], mask.astype(np.float32))]
    while min(levels[-1][1].shape) > 1:
        c, w = levels[-1]
        h2, w2 = c.shape[0] // 2, c.shape[1] // 2
        levels.append((c[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, 3).sum((1, 3)),
                       w[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).sum((1, 3))))
    c, w = levels[-1]
    fill = c / np.maximum(w, 1e-9)[..., None]
    for c, w in reversed(levels[:-1]):
        up = np.repeat(np.repeat(fill, 2, 0), 2, 1)
        if up.shape[:2] != c.shape[:2]:
            up = np.pad(up, ((0, max(0, c.shape[0] - up.shape[0])),
                             (0, max(0, c.shape[1] - up.shape[1])), (0, 0)), mode='edge')
            up = up[:c.shape[0], :c.shape[1]]
        fill = np.where((w > 0)[..., None], c / np.maximum(w, 1e-9)[..., None], up)
    out = img.copy()
    out[~mask] = fill[~mask]
    return out


def label(mask):
    """connected regions of a boolean image (4-neighbour) -> labels, count"""
    try:
        from scipy import ndimage
        return ndimage.label(mask)
    except ImportError:
        pass
    # runs of each row, joined to the runs they touch in the row above
    h, w = mask.shape
    lab = np.zeros(mask.shape, np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    prev = []
    for yy in range(h):
        row = mask[yy]
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        cur = []
        k = 0
        for s0, e0 in zip(starts, ends):
            parent.append(len(parent))
            me = len(parent) - 1
            while k < len(prev) and prev[k][1] <= s0:
                k += 1
            kk = k
            while kk < len(prev) and prev[kk][0] < e0:
                ra, rb = find(me), find(prev[kk][2])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
                kk += 1
            cur.append((s0, e0, me))
            lab[yy, s0:e0] = me
        prev = cur
    roots = np.array([find(i) for i in range(len(parent))])
    _, comp = np.unique(roots, return_inverse=True)
    return comp[lab].astype(np.int32), int(comp.max())


# ---- the steps -----------------------------------------------------------------
def page_nodes(j):
    """{node index: (page, size)} for the light page nodes"""
    out = {}
    for ni, n in enumerate(j['nodes']):
        m = PAGE_RE.match(n.get('name', ''))
        if m and 'mesh' in n:
            out[ni] = (int(m.group(1)), int(m.group(2)))
    return out


def step_scan(a, W):
    g = Glb(a.glb or os.path.join(a.bake, 'map.glb'))
    pn = page_nodes(g.j)
    if not pn:
        raise SystemExit('no light page nodes (lmNNN_zlms-N) in map.glb')
    if any(re.match(r'^lm\d+n\d*_zlms-\d+$', n.get('name', '')) for n in g.j['nodes']):
        # never-solid page nodes (bake_glb: lmNNNn_zlms-N, each with a
        # -collider stand-in) are not carried through a relayout yet
        raise SystemExit('this map has never-solid light page nodes (lmNNNn_zlms-N): '
                         'bake it again with bake_glb.py rather than relayout it')
    sizes = {s for _, s in pn.values()}
    if len(sizes) != 1:
        raise SystemExit(f'pages of more than one size: {sizes}')
    S = sizes.pop()
    prims, rows = [], []
    for ni, (page, _) in sorted(pn.items()):
        for pi, pr in enumerate(g.j['meshes'][g.j['nodes'][ni]['mesh']]['primitives']):
            at = pr['attributes']
            if 'TEXCOORD_1' not in at:
                raise SystemExit(f'{g.j["nodes"][ni]["name"]}: a primitive with no light UVs')
            if 'indices' in pr:
                idx = g.acc(pr['indices'])
                if not np.array_equal(idx, np.arange(len(idx))):
                    raise SystemExit('indexed light page primitives: not a bake_glb --zone-v3 map')
            P = g.acc(at['POSITION']).astype(np.float32).reshape(-1, 3, 3)
            U = g.acc(at['TEXCOORD_1']).astype(np.float64).reshape(-1, 3, 2) * S
            lo = np.floor(U.min(1)).astype(np.int32)
            hi = np.ceil(U.max(1)).astype(np.int32)
            hi = np.maximum(hi, lo + 1)
            k = len(prims)
            prims.append((ni, pi, page, int(pr.get('material', -1))))
            rows.append(dict(prim=np.full(len(P), k, np.int32), cen=P.mean(1), box=np.concatenate([lo, hi], 1)))
    cat = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
    np.savez(os.path.join(W, 'scan.npz'), prims=np.array(prims, np.int64), size=S, **cat)
    print(f'scan: {len(pn)} light page nodes, {len(prims)} primitives, {len(cat["prim"]):,} triangles, pages of {S}')


def step_charts(a, W):
    d = np.load(os.path.join(W, 'scan.npz'))
    prims, S = d['prims'], int(d['size'])
    tp, cen, box = d['prim'], d['cen'], d['box']
    tri_page = prims[tp, 2]
    tri_mat = prims[tp, 3]
    chart = np.full(len(tp), -1, np.int64)
    c_page, c_box, c_cen, c_mat, c_n = [], [], [], [], 0
    for page in np.unique(tri_page):
        sel = np.nonzero(tri_page == page)[0]
        b = box[sel]
        # every triangle's texels, one texel wider, marked; the charts are the
        # separate patches (a chart has PAD texels on each side before the next)
        diff = np.zeros((S + 2, S + 2), np.int32)
        x0 = np.clip(b[:, 0] - 1, 0, S); y0 = np.clip(b[:, 1] - 1, 0, S)
        x1 = np.clip(b[:, 2] + 1, 0, S); y1 = np.clip(b[:, 3] + 1, 0, S)
        np.add.at(diff, (y0, x0), 1); np.add.at(diff, (y0, x1), -1)
        np.add.at(diff, (y1, x0), -1); np.add.at(diff, (y1, x1), 1)
        cover = diff.cumsum(0).cumsum(1)[:S, :S] > 0
        lab, n = label(cover)
        cy = np.clip((b[:, 1] + b[:, 3]) // 2, 0, S - 1)
        cx = np.clip((b[:, 0] + b[:, 2]) // 2, 0, S - 1)
        l = lab[cy, cx].astype(np.int64)
        if (l == 0).any():
            raise SystemExit(f'page {page}: a triangle outside every chart')
        # 1..n -> chart ids, only those with triangles
        used, inv = np.unique(l, return_inverse=True)
        ids = c_n + inv
        chart[sel] = ids
        m = len(used)
        bx = np.full((m, 4), [S, S, 0, 0], np.int64)
        np.minimum.at(bx[:, 0], inv, b[:, 0]); np.minimum.at(bx[:, 1], inv, b[:, 1])
        np.maximum.at(bx[:, 2], inv, b[:, 2]); np.maximum.at(bx[:, 3], inv, b[:, 3])
        cs = np.zeros((m, 3)); np.add.at(cs, inv, cen[sel].astype(np.float64))
        cnt = np.bincount(inv, minlength=m)
        cm = np.full(m, -1, np.int64); cm[inv] = tri_mat[sel]
        c_page.append(np.full(m, page)); c_box.append(bx); c_cen.append(cs / cnt[:, None]); c_mat.append(cm)
        c_n += m
    c_box = np.concatenate(c_box)
    # a chart's texels with their padding stay on their old page
    if (c_box[:, :2] - PL.PAD < 0).any() or (c_box[:, 2:] + PL.PAD > S).any():
        print('charts: some charts touch the page edge (their padding is clipped)')
    np.savez(os.path.join(W, 'charts.npz'), chart=chart, page=np.concatenate(c_page), box=c_box,
             cen=np.concatenate(c_cen), mat=np.concatenate(c_mat), size=S)
    tw = c_box[:, 2] - c_box[:, 0]; th = c_box[:, 3] - c_box[:, 1]
    print(f'charts: {c_n:,} charts on {len(np.unique(tri_page))} pages, {((tw + 4) * (th + 4)).sum() / 1e6:.1f} M texels')
    # the layout, by location
    newp, nx, ny, npg = PL.layout(np.concatenate(c_cen), tw, th, np.concatenate(c_mat), S)
    np.savez(os.path.join(W, 'layout.npz'), page=newp, x=nx, y=ny, pages=npg)
    print(f'layout: {npg} pages by location (was {len(np.unique(tri_page))})')


def step_light(a, W, deadline):
    c = np.load(os.path.join(W, 'charts.npz')); L = np.load(os.path.join(W, 'layout.npz'))
    S = int(c['size']); npg = int(L['pages'])
    info = json.load(open(os.path.join(a.bake, 'bake_info.json')))
    lmname = {}
    for o in info['objects']:
        m = PAGE_RE.match(o['name'])
        if m:
            lmname.setdefault(int(m.group(1)), o.get('lightmap', o['name']))
    src_dir = os.path.join(a.bake, 'lightmaps')
    out_dir = os.path.join(W, 'lightmaps')
    os.makedirs(out_dir, exist_ok=True)
    # new pages done already (a resumed run) are skipped
    todo = [p for p in range(npg) if not os.path.isfile(os.path.join(out_dir, f'lm{p:03d}_zlms-{S}.exr'))]
    if not todo:
        print('light: all pages written'); return True
    pad = PL.PAD
    old_page, box = c['page'], c['box']
    newp, nx, ny = L['page'], L['x'], L['y']
    # a batch of new pages at a time, each filled from the old pages it draws on
    batch = todo[:max(1, a.batch)]
    imgs = {p: np.zeros((S, S, 3), np.float32) for p in batch}
    mask = {p: np.zeros((S, S), bool) for p in batch}
    need = np.isin(newp, batch)
    for op in np.unique(old_page[need]):
        src = read_exr(os.path.join(src_dir, lmname.get(int(op), f'lm{int(op):03d}_zlms-{S}') + '.exr'))
        for i in np.nonzero(need & (old_page == op))[0]:
            x0, y0, x1, y1 = box[i]
            sx0, sy0 = max(x0 - pad, 0), max(y0 - pad, 0)
            sx1, sy1 = min(x1 + pad, S), min(y1 + pad, S)
            dx, dy = int(nx[i]) - x0, int(ny[i]) - y0
            p = int(newp[i])
            imgs[p][sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = src[sy0:sy1, sx0:sx1]
            mask[p][sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = True
    for p in batch:
        write_exr_half(os.path.join(out_dir, f'lm{p:03d}_zlms-{S}.exr'), fill_empty(imgs[p], mask[p]))
    left = len(todo) - len(batch)
    print(f'light: pages {batch[0]}-{batch[-1]} written, {left} to go')
    return left == 0


def step_glb(a, W):
    g = Glb(a.glb or os.path.join(a.bake, 'map.glb'))
    d = np.load(os.path.join(W, 'scan.npz')); c = np.load(os.path.join(W, 'charts.npz'))
    L = np.load(os.path.join(W, 'layout.npz'))
    S = int(d['size']); npg = int(L['pages'])
    prims = d['prims']; tp = d['prim']; chart = c['chart']
    box = c['box']
    shift = np.stack([L['x'] - box[:, 0], L['y'] - box[:, 1]], 1).astype(np.float64) / S   # per chart, in UV
    tri_new = L['page'][chart]
    tri_mat = prims[tp, 3]
    prim_start = np.searchsorted(tp, np.arange(len(prims)))    # the scan keeps each primitive's triangles together
    j = g.j
    pn = page_nodes(j)
    # what stays: every node but the old pages, and what they use
    keep_nodes = [i for i in range(len(j['nodes'])) if i not in pn]
    keep_meshes = sorted({j['nodes'][i]['mesh'] for i in keep_nodes if 'mesh' in j['nodes'][i]})
    keep_acc = sorted({v for mi in keep_meshes for pr in j['meshes'][mi]['primitives']
                       for v in list(pr['attributes'].values()) + ([pr['indices']] if 'indices' in pr else [])})
    keep_views = sorted({j['accessors'][ai]['bufferView'] for ai in keep_acc if 'bufferView' in j['accessors'][ai]}
                        | {im['bufferView'] for im in j.get('images', []) if 'bufferView' in im})
    tmp = os.path.join(W, 'bin.part')
    nj = {k: v for k, v in j.items() if k not in ('nodes', 'meshes', 'accessors', 'bufferViews', 'buffers', 'scenes')}
    nj['bufferViews'], nj['accessors'], nj['meshes'], nj['nodes'] = [], [], [], []
    with open(tmp, 'wb') as fo:
        pos = [0]

        def put(data, target=None):
            pad = (-pos[0]) % 4
            fo.write(b'\0' * pad)
            pos[0] += pad
            v = {'buffer': 0, 'byteOffset': pos[0], 'byteLength': len(data)}
            if target:
                v['target'] = target
            fo.write(data)
            pos[0] += len(data)
            nj['bufferViews'].append(v)
            return len(nj['bufferViews']) - 1
        vmap = {}
        for vi in keep_views:
            ov = j['bufferViews'][vi]
            nv = put(bytes(g.view(vi)), ov.get('target'))
            for k in ('byteStride', 'name'):
                if k in ov:
                    nj['bufferViews'][nv][k] = ov[k]
            vmap[vi] = nv
        amap = {}
        for ai in keep_acc:
            acc = dict(j['accessors'][ai])
            if 'bufferView' in acc:
                acc['bufferView'] = vmap[acc['bufferView']]
            nj['accessors'].append(acc)
            amap[ai] = len(nj['accessors']) - 1
        for im in nj.get('images', []):
            if 'bufferView' in im:
                im['bufferView'] = vmap[im['bufferView']]
        mmap_ = {}
        for mi in keep_meshes:
            me = json.loads(json.dumps(j['meshes'][mi]))
            for pr in me['primitives']:
                pr['attributes'] = {k: amap[v] for k, v in pr['attributes'].items()}
                if 'indices' in pr:
                    pr['indices'] = amap[pr['indices']]
            nj['meshes'].append(me)
            mmap_[mi] = len(nj['meshes']) - 1

        def acc_of(arr, ctype, typ, target, minmax=False):
            arr = np.ascontiguousarray(arr)
            v = put(arr.tobytes(), target)
            ad = {'bufferView': v, 'componentType': ctype, 'count': int(arr.shape[0]), 'type': typ}
            if minmax:
                ad['min'] = [float(x) for x in arr.min(0)]
                ad['max'] = [float(x) for x in arr.max(0)]
            nj['accessors'].append(ad)
            return len(nj['accessors']) - 1
        # the new pages: their triangles, by material, from wherever they were
        order = np.argsort(tri_new, kind='stable')
        bounds = np.searchsorted(tri_new[order], np.arange(npg + 1))
        new_nodes = []
        ext = []
        for p in range(npg):
            sel = order[bounds[p]:bounds[p + 1]]
            if not len(sel):
                continue
            prs = []
            lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
            for m in np.unique(tri_mat[sel]):
                sm = sel[tri_mat[sel] == m]
                parts = {'POSITION': [], 'NORMAL': [], 'TEXCOORD_0': [], 'TEXCOORD_1': []}
                for k in np.unique(tp[sm]):
                    rows_t = sm[tp[sm] == k]
                    local = rows_t - prim_start[k]            # triangle numbers in that primitive
                    vr = (local[:, None] * 3 + np.arange(3)).reshape(-1)
                    ni, pi = int(prims[k, 0]), int(prims[k, 1])
                    at = j['meshes'][j['nodes'][ni]['mesh']]['primitives'][pi]['attributes']
                    for key in parts:
                        if key in at:
                            arr = g.acc(at[key], vr).astype(np.float32)
                            if key == 'TEXCOORD_1':
                                arr = arr + np.repeat(shift[chart[rows_t]], 3, 0).astype(np.float32)
                            parts[key].append(arr)
                attrs = {}
                P = np.concatenate(parts['POSITION'])
                lo = np.minimum(lo, P.min(0)); hi = np.maximum(hi, P.max(0))
                attrs['POSITION'] = acc_of(P, 5126, 'VEC3', 34962, True)
                if parts['NORMAL']:
                    attrs['NORMAL'] = acc_of(np.concatenate(parts['NORMAL']), 5126, 'VEC3', 34962)
                if parts['TEXCOORD_0']:
                    attrs['TEXCOORD_0'] = acc_of(np.concatenate(parts['TEXCOORD_0']), 5126, 'VEC2', 34962)
                attrs['TEXCOORD_1'] = acc_of(np.concatenate(parts['TEXCOORD_1']), 5126, 'VEC2', 34962)
                ii = acc_of(np.arange(len(P), dtype='<u4'), 5125, 'SCALAR', 34963)
                pr = {'attributes': attrs, 'indices': ii, 'mode': 4}
                if m >= 0:
                    pr['material'] = int(m)
                prs.append(pr)
            ext.append(hi - lo)
            # Godot's 256 surfaces a mesh: more materials than that, more nodes
            parts_n = -(-len(prs) // PL.MAX_SURFACES)
            size_ = -(-len(prs) // parts_n)
            for kk in range(parts_n):
                name = f'lm{p:03d}_zlms-{S}' if kk == 0 else f'lm{p:03d}s{kk}_zlms-{S}'
                nj['meshes'].append({'name': name, 'primitives': prs[kk * size_:(kk + 1) * size_]})
                new_nodes.append((p, name, len(nj['meshes']) - 1))
        total = pos[0] + ((-pos[0]) % 4)
        fo.write(b'\0' * ((-pos[0]) % 4))
    # nodes: the kept ones (children remapped), then the new pages at the root
    nmap = {old: k for k, old in enumerate(keep_nodes)}
    for old in keep_nodes:
        n = json.loads(json.dumps(j['nodes'][old]))
        if 'mesh' in n:
            n['mesh'] = mmap_[n['mesh']]
        if 'children' in n:
            n['children'] = [nmap[ch] for ch in n['children'] if ch in nmap]
            if not n['children']:
                del n['children']
        nj['nodes'].append(n)
    roots = [nmap[r] for r in j['scenes'][j.get('scene', 0)]['nodes'] if r in nmap]
    for p, name, mi in new_nodes:
        nj['nodes'].append({'name': name, 'mesh': mi})
        roots.append(len(nj['nodes']) - 1)
    nj['scenes'] = [{'nodes': roots}]
    nj['scene'] = 0
    nj['buffers'] = [{'byteLength': total}]
    js = json.dumps(nj, separators=(',', ':')).encode('utf-8')
    js += b' ' * ((-len(js)) % 4)
    out = os.path.join(W, 'map.glb')
    with open(out, 'wb') as fo, open(tmp, 'rb') as fi:
        fo.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + total))
        fo.write(struct.pack('<II', len(js), 0x4E4F534A)); fo.write(js)
        fo.write(struct.pack('<II', total, 0x004E4942))
        shutil.copyfileobj(fi, fo, 64 << 20)
    os.remove(tmp)
    json.dump({'objects': [{'name': name, 'size': S, **({} if name == f'lm{p:03d}_zlms-{S}'
                            else {'lightmap': f'lm{p:03d}_zlms-{S}'})} for p, name, _ in new_nodes],
               'extent_median': [float(v) for v in np.median(np.array(ext), 0)]},
              open(os.path.join(W, 'objects.json'), 'w'))
    print(f'glb: {len(new_nodes)} light page nodes, spanning a median '
          f'{" x ".join(str(int(v)) for v in np.median(np.array(ext), 0))} m each -> {out}')


def step_finish(a, W):
    """bake_info.json and occluder.bin into the work folder"""
    info = json.load(open(os.path.join(a.bake, 'bake_info.json')))
    ob = json.load(open(os.path.join(W, 'objects.json')))
    info['objects'] = ob['objects']
    info['reuse_textures'] = True
    json.dump(info, open(os.path.join(W, 'bake_info.json'), 'w'), indent=1)
    if a.source:
        src = Glb(a.source)
        ov, oi, k = OCC.from_glb(src.j, src.acc)
    else:
        g = Glb(os.path.join(W, 'map.glb'))
        ov, oi, k = OCC.from_glb(g.j, g.acc, skip=None)
    OCC.write(os.path.join(W, 'occluder.bin'), ov, oi)
    print(f'finish: bake_info.json with {len(ob["objects"])} light objects; occluder.bin with '
          f'{len(oi) // 3:,} triangles ({k.get("world", 0):,} of walls and floors, {k.get("sky", 0):,} of sky)')


def install(a, W):
    """the new files into the bake folder, the old ones kept beside them"""
    b = a.bake
    for name in ('map.glb', 'bake_info.json', 'occluder.bin', 'lightmaps'):
        src, keep = os.path.join(b, name), os.path.join(b, name.replace('.', '_before_relayout.', 1)
                                                         if '.' in name else name + '_before_relayout')
        if os.path.exists(src) and not os.path.exists(keep):
            os.replace(src, keep)
    for name in ('map.glb', 'bake_info.json', 'occluder.bin'):
        shutil.copy2(os.path.join(W, name), os.path.join(b, name))
    shutil.copytree(os.path.join(W, 'lightmaps'), os.path.join(b, 'lightmaps'))
    shutil.copy2(os.path.join(HERE, 'pack_zone_map.gd'), os.path.join(b, 'pack_zone_map.gd'))
    print(f'installed into {b} (the old map.glb, bake_info.json, occluder.bin and lightmaps/ kept as *_before_relayout)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bake', help='the bake folder (<map>_v3_zone_v3)')
    ap.add_argument('--source', help="the forge's .glb the bake was made from, for the occluders "
                                     "(default: the bake folder's parent's <map>.glb if there is one)")
    ap.add_argument('--work', help='where the steps keep their results')
    ap.add_argument('--step', choices=['scan', 'charts', 'light', 'glb', 'finish', 'install'],
                    help='run one step (default: all of them)')
    ap.add_argument('--batch', type=int, default=8, help='new light pages made at a time (memory)')
    ap.add_argument('--glb', help="read the bake's map.glb from here instead (a copy on a local disk)")
    a = ap.parse_args()
    a.bake = os.path.abspath(a.bake)
    if a.source is None:
        parent = os.path.dirname(a.bake)
        cand = os.path.join(parent, os.path.basename(parent) + '.glb')
        a.source = cand if os.path.isfile(cand) else None
    W = a.work or a.bake.rstrip('/\\') + '_relayout'
    os.makedirs(W, exist_ok=True)
    t0 = time.time()
    steps = [a.step] if a.step else ['scan', 'charts', 'light', 'glb', 'finish', 'install']
    for s in steps:
        if s == 'light':
            while not step_light(a, W, None) and not a.step:
                pass
        else:
            {'scan': step_scan, 'charts': step_charts, 'glb': step_glb, 'finish': step_finish,
             'install': install}[s](a, W)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
