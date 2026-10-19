"""Bring a map an earlier forge.py made up to date with what the forge now
carries for the map_extras mod - its 3D skybox room, its teleports, its prop
doors open - without converting or baking it again.

    python tools/add_extras.py build  <map.gma|map.bsp> <convert_report.json> <work dir> [--game-dir D]
    python tools/add_extras.py doors  <map.gma|map.bsp> <convert_report.json> <work dir> [--game-dir D]
    python tools/add_extras.py patch  <work dir> <v3 map.glb> [<v3 map.glb> ...] [--rewrite]
    python tools/add_extras.py undo   <v3 map.glb>

build   the 3D skybox room (its brushwork and props, lit with the map's own
        light), the teleports and the room's anchor, as the forge makes them
        now (extras.py), in the map's own space - read from the report's
        origin_shift and map_turn_deg - into <work>/extras.glb
doors   the map's prop_door_rotating doors, each where the bake has it
        (closed) and how far it turns to open about its hinge, and the props
        its logic animates open (left out now): <work>/doors.npz
patch   a v3 map folder's map.glb, in place: the door triangles in the light
        pages turned open (or, for the animated props, folded to nothing),
        and extras.glb's nodes added. Only the bytes that change are written
        - the vertices of the doors, the JSON, what is added at the end - so
        a 1 GB map takes seconds, and the originals of all of it go to
        <map.glb>.extras_undo first: `undo` puts them back.

A door keeps the light it was baked with while shut, and the floor beyond it
the light it had with the door shut: a full forge + bake lights them open.
"""
import argparse, json, math, os, sys, struct, collections, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from zonemap import bsp as B, build as BU, gma as GMA, locate, extras as EX, sky as SKY
from zonemap import flatten as FL, propbuild as PB, convert as CV
from zonemap.sourcefs import SourceFs
from zonemap.glb import Glb
from zonemap.materials import MaterialBank

GLB_MAGIC, JSON_CT, BIN_CT = 0x46546C67, 0x4E4F534A, 0x004E4942


# --------------------------------------------------------------- the source
def open_map(src, game_dir=None):
    archives, data = [], None
    if src.lower().endswith('.gma'):
        arc = GMA.Gma(src)
        archives.append(arc)
        maps = sorted(arc.maps(), key=lambda m: -m[1])
        data = arc.read(maps[0][0])
        path = os.path.join(os.path.dirname(os.path.abspath(src)), os.path.basename(maps[0][0]))
        try:
            dirs = locate.find_game('gmod', game_dir)[1]
        except SystemExit:
            dirs = []
    else:
        path = os.path.abspath(src)
        dirs = locate.content_roots_for_bsp(path, locate.guess_game(path) or 'gmod', game_dir)
    bsp = B.Bsp(path, data=data)
    fs = SourceFs(dirs, pakfile_bytes=bsp.lump(B.L_PAKFILE), archives=archives)
    return bsp, fs, os.path.splitext(os.path.basename(path))[0]


class Frame:
    """the map's own space, as the report says the forge made it"""

    def __init__(self, rep):
        self.scale = float(rep.get('scale_m_per_unit', BU.UNIT))
        self.shift = np.asarray(rep.get('origin_shift', [0, 0, 0]), np.float64)
        self.turn = math.radians(float(rep.get('map_turn_deg', 0.0)))
        t = self.turn
        self.R = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
        self.q = [0.0, math.sin(t / 2.0), 0.0, math.cos(t / 2.0)]

    def to_map(self, P):
        return (BU.to_gltf(np.asarray(P, np.float64).reshape(-1, 3), self.scale) + self.shift) @ self.R.T

    def dir_to_map(self, d):
        return BU.to_gltf(np.asarray(d, np.float64)[None, :], 1.0)[0] @ self.R.T

    def to_src(self, P):
        g = np.asarray(P, np.float64).reshape(-1, 3) @ self.R - self.shift
        return np.stack([g[:, 0], -g[:, 2], g[:, 1]], axis=1) / self.scale


def _bank(glb, fs, mapname):
    # as forge.py makes them by default
    return MaterialBank(glb, fs, mapname, albedo_px=512, normal_px=256, albedo_jpeg=True,
                        jpeg_q=90, gen_normals=True, flip_green=False, emissive=True,
                        metal_sheen=0.15, detail=True, normal_strength=1.0,
                        roughness_scale=1.0, roughness_min=0.0, water_roughness=0.6,
                        water_value=0.28)


def _rename(glb):
    seen = {}
    for mm in glb.j['materials']:
        if mm['name'].startswith('z_'):
            continue
        if not mm['name'].startswith('src_'):
            mm['name'] = 'src_' + CV.slug(mm['name'])
        if mm['name'] in seen:
            seen[mm['name']] += 1
            mm['name'] = f"{mm['name']}_{seen[mm['name']]}"
        else:
            seen[mm['name']] = 0


# ------------------------------------------------------------------- build
def cmd_build(a):
    rep = json.load(open(a.report))
    F = Frame(rep)
    bsp, fs, mapname = open_map(a.map, a.game_dir)
    scale = F.scale
    sc = CV.sky_camera(bsp)
    play = CV.playable_centre(bsp)
    cut = float((rep.get('geometry_stats') or {}).get('skybox_cut', CV.SKYBOX_BIAS))
    glb = Glb(generator='zone-map-forge add_extras')
    kids, sky3d_nodes, sky3d_sky = [], [], None
    from zonemap.materials import blend_band_count
    if sc is not None and play is not None and len(play):
        sky_g = BU.to_gltf(sc[None, :], scale)[0]
        play_pts = BU.to_gltf(play, scale)

        def in_room(c_src):
            return CV.in_3d_skybox(BU.to_gltf(np.asarray(c_src)[None, :], scale)[0], sky_g, play_pts, cut)
        bank = _bank(glb, fs, mapname)
        chunks, st = BU.extract(bsp, scale=scale, cell_m=48.0, fill_mode='none', doors='open',
                                blend_bands=lambda m: blend_band_count(fs, m, mapname, 5),
                                face_filter=in_room, verbose=False)
        chunks = [c for c in chunks if CV.in_3d_skybox(c[2].mean(axis=0), sky_g, play_pts, cut)]
        shift32 = F.shift.astype(np.float32)
        mat_idx = {}
        for m, cell, pos, nrm, uv, idx in chunks:
            if m not in mat_idx:
                mat_idx[m] = (glb.material(m, roughness=0.9) if m.startswith('z_') else bank.get(m))
            mid = mat_idx[m]
            pos = pos + shift32
            if mid in bank.two_sided:
                pos, nrm, uv, idx = CV.mirror_faces(pos, nrm, uv, idx)
            nm = f'{EX.SKY3D}_{CV.slug(m)}_{cell[0]}_{cell[1]}_{cell[2]}'
            kids.append(glb.node(nm, mesh=glb.mesh(nm, [(pos, nrm, uv, idx, mid)])))
            sky3d_nodes.append((kids[-1], m))
        sg_s, pp_s = sky_g + F.shift, play_pts + F.shift

        def sky3d_region(t, _s=sg_s, _p=pp_s):
            return CV.in_3d_skybox(np.asarray(t, np.float64), _s, _p, cut)
        pnodes, pst = PB.add_props(glb, fs, bank, bsp, scale=scale, doors='open',
                                   drop_targets=EX.animated_open(bsp),
                                   keep_region=lambda t: False, alt_region=sky3d_region,
                                   alt_prefix=EX.SKY3D + '_', only_alt=True,
                                   carriers=BU.door_carriers(bsp, doors='open'),
                                   offset=F.shift, verbose=False)
        kids += pnodes
        _rename(glb)
        sp, sn = SKY.faces(bsp, scale)
        if len(sp):
            cen = sp.reshape(-1, 3, 3).mean(axis=1)
            sky3d_sky = SKY.keep_tris(sp, sn, [CV.in_3d_skybox(c, sky_g, play_pts, cut) for c in cen])[0]
            sky3d_sky = (np.asarray(sky3d_sky, np.float64) + F.shift) @ F.R.T
        print(f'build: the 3D skybox room: {sum(len(c[5]) for c in chunks):,} triangles in {len(chunks)} '
              f'chunks, {len(pnodes)} props ({dict(pst)})')
    root = glb.node(mapname + '_extras', children=kids, rotation=F.q if F.turn else None)
    glb.root(root)
    if F.turn:
        FL.flatten_roots(glb.j, glb.blob)
    tps = EX.teleports(bsp, F.to_map, F.dir_to_map)
    for i in EX.add_teleports(glb, tps):
        glb.root(i)
    info = {'teleports': len(tps)}
    if sky3d_nodes and sc is not None:
        anchor = F.to_map(sc)[0]
        glb.root(EX.add_sky3d(glb, anchor, F.to_map(np.zeros(3))[0], CV.sky_camera_scale(bsp)))
        props3d = [i for i, n in enumerate(glb.j['nodes'])
                   if str(n.get('name', '')).startswith(EX.SKY3D + '_prop') and 'mesh' in n]
        lst, seen = EX.light_sky3d(glb, bsp, sky3d_nodes, props3d, F.to_src, None, anchor,
                                   unlit=EX.unlit_materials(bsp), sky_tris=sky3d_sky,
                                   sky_colour=SKY.COLOUR, verbose=True)
        info.update({'sky3d': {'anchor': anchor.round(3).tolist(), 'scale': CV.sky_camera_scale(bsp),
                               'light': lst}, 'sky_colour': None if seen is None else [float(c) for c in seen]})
    os.makedirs(a.work, exist_ok=True)
    out = os.path.join(a.work, 'extras.glb')
    n = glb.save(out)
    json.dump(info, open(os.path.join(a.work, 'extras.json'), 'w'), indent=1)
    print(f'build: {out} ({n / 1e6:.1f} MB): {len(tps)} teleports '
          f'({sum(1 for t in tps if t["start_disabled"])} start switched off); {json.dumps(info.get("sky_colour"))} '
          f'the sky surfaces\' plain colour')


# ------------------------------------------------------------------- doors
def _posed_models(fs, paths):
    """the models an older forge drew in the wrong pose (mdlpose.py): read
    unposed and posed, the vertices differ"""
    from zonemap import mdl as MD, mdlpose as MP
    out = set()
    for path in paths:
        p = path.lower()
        p = p if p.startswith('models/') else 'models/' + p
        base = p[:-4] if p.endswith('.mdl') else p
        mb, vb = fs.read(base + '.mdl'), fs.read(base + '.vvd')
        if mb is None or vb is None:
            continue
        try:
            m = MD.Mdl(mb)
            if m.is_staticprop:
                continue
            pose, _ = MP.frame0(mb, read=lambda: fs.read(base + '.ani'))
            if pose is not None and MP.skin(MD.Vvd(vb), pose)[2] > 0.01:
                out.add(path.lower())
        except Exception:
            continue
    return out


def _world_tris(glb, ni):
    J = glb.j
    n = J['nodes'][ni]
    M = FL.node_matrix(n)
    T = []
    for pr in J['meshes'][n['mesh']]['primitives']:
        pos = EX._read(glb, pr['attributes']['POSITION']).astype(np.float64)
        idx = EX._read(glb, pr['indices']).reshape(-1) if 'indices' in pr else np.arange(len(pos))
        T.append((pos @ M[:3, :3].T + M[:3, 3])[idx.reshape(-1, 3)])
    return np.concatenate(T) if T else np.zeros((0, 3, 3))


def cmd_doors(a):
    """what the props an older forge placed need: the prop doors opened (not
    the ones locked for good), the props the map's logic animates open left
    out, and every prop whose model it drew in the wrong pose (doors and
    switches built a quarter turn from their idle sequence: mdlpose.py) put
    right - each as its triangles where the bake has them, and where they go"""
    from zonemap import mdl as MD
    rep = json.load(open(a.report))
    F = Frame(rep)
    bsp, fs, mapname = open_map(a.map, a.game_dir)
    scale = F.scale
    drop = EX.animated_open(bsp)
    logic = EX.door_logic(bsp)
    pl, _n, _m = PB.collect(bsp, True, doors='closed')
    posed = _posed_models(fs, {p[0].lower() for p in pl})
    doors, drops = {}, {}
    shut = 0
    for e in bsp.entities:
        cn = e.get('classname', '')
        tn = str(e.get('targetname', '')).lower()
        try:
            o = tuple(round(float(x), 2) for x in e.get('origin', '0 0 0').split()[:3])
        except ValueError:
            continue
        if cn == 'prop_door_rotating':
            if EX.door_stays_shut(e, logic):
                shut += 1
            else:
                doors[(cn, o)] = e
        elif cn.startswith('prop_dynamic') and tn in drop:
            drops[(cn, o)] = e

    def key(pl_):
        return (pl_[5], tuple(round(float(x), 2) for x in pl_[2]))

    def select(pl_):
        return pl_[0].lower() in posed or key(pl_) in doors or key(pl_) in drops
    sc = CV.sky_camera(bsp)
    play = CV.playable_centre(bsp)
    cut = float((rep.get('geometry_stats') or {}).get('skybox_cut', CV.SKYBOX_BIAS))
    region = None
    if sc is not None and play is not None and len(play):
        sg = BU.to_gltf(sc[None, :], scale)[0] + F.shift
        pp = BU.to_gltf(play, scale) + F.shift

        def region(t, _s=sg, _p=pp):
            return not CV.in_3d_skybox(np.asarray(t, np.float64), _s, _p, cut)
    runs = []
    for pose, doors_mode in ((False, 'closed'), (True, 'open')):
        MD.POSE = pose
        glb = Glb(generator='zone-map-forge add_extras doors')
        bank = _bank(glb, fs, mapname)
        nodes, pst = PB.add_props(glb, fs, bank, bsp, scale=scale, doors=doors_mode, select=select,
                                  keep_region=region, carriers=BU.door_carriers(bsp, doors='open'),
                                  offset=F.shift, verbose=False)
        root = glb.node('props', children=nodes, rotation=F.q if F.turn else None)
        glb.root(root)
        if F.turn:
            FL.flatten_roots(glb.j, glb.blob)
        runs.append((glb, nodes, pst))
    MD.POSE = True
    (go, no, _), (gn, nn, pst) = runs
    if len(no) != len(nn):
        raise SystemExit(f'the two readings placed {len(no)} and {len(nn)} props')
    old_t, new_t, meta = [], [], []
    for i_o, i_n in zip(no, nn):
        To, Tn = _world_tris(go, i_o), _world_tris(gn, i_n)
        node = go.j['nodes'][i_o]
        t = FL.node_matrix(node)[:3, 3]
        o = F.to_src(t[None, :])[0]
        k = min(list(doors) + list(drops), key=lambda kk: np.linalg.norm(np.array(kk[1]) - o), default=None)
        is_drop = k is not None and k in drops and np.linalg.norm(np.array(k[1]) - o) < 1.0
        if is_drop:
            action = 'drop'
        else:
            if To.shape != Tn.shape:
                print(f'doors: {node.get("name")}: {len(To)} triangles read one way, {len(Tn)} the other - left')
                continue
            if np.abs(To - Tn).max() < 1e-4:
                continue
            action = 'move'
        old_t.append(To)
        new_t.append(Tn if action == 'move' else To)
        what = ('door' if k is not None and k in doors and np.linalg.norm(np.array(k[1]) - o) < 1.0 else 'posed')
        meta.append({'name': node.get('name'), 'origin': [round(float(x), 1) for x in o], 'action': action,
                     'kind': 'animated' if action == 'drop' else what, 'triangles': int(len(To))})
    os.makedirs(a.work, exist_ok=True)
    np.savez_compressed(os.path.join(a.work, 'doors.npz'),
                        **{f't{i}': t.astype(np.float32) for i, t in enumerate(old_t)},
                        **{f'n{i}': t.astype(np.float32) for i, t in enumerate(new_t)})
    json.dump({'doors': meta, 'posed_models': sorted(posed), 'locked_shut': shut},
              open(os.path.join(a.work, 'doors.json'), 'w'), indent=1)
    c = collections.Counter(m['kind'] for m in meta)
    print(f'doors: {c.get("door", 0)} prop doors opened ({shut} locked for good stay shut), '
          f'{c.get("animated", 0)} animated props left out, {c.get("posed", 0)} more props of '
          f'{len(posed)} models put in their pose; {sum(m["triangles"] for m in meta):,} triangles')


# ------------------------------------------------------------------- patch
import re
PAGE_RE = re.compile(r'^lm\d+(s\d+)?_zlms-\d+$')
_CT = {5126: ('<f4', 4), 5125: ('<u4', 4), 5123: ('<u2', 2), 5121: ('u1', 1)}
_NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
OFF = 1 << 20


def _readn(f, n):
    """n bytes, or an error: a read on a network or VM share can come back
    short (it did, once, for 2 of 31 MB) - which would quietly write a
    broken map"""
    out = bytearray()
    while len(out) < n:
        c = f.read(n - len(out))
        if not c:
            raise IOError(f'{getattr(f, "name", "file")}: ended {n - len(out)} bytes short')
        out += c
    return bytes(out)


def _keys(P):
    q = np.round(np.asarray(P, np.float64).reshape(-1, 3) * 1000.0).astype(np.int64) + OFF
    if (q < 0).any() or (q >= 1 << 21).any():
        raise SystemExit('a vertex more than 1 km from the origin: the key encoding cannot hold it')
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


class GlbFile:
    """a .glb read in place: its JSON, and accessors read with seek + read"""

    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        h = _readn(self.f, 20)
        magic, ver, total = struct.unpack('<III', h[:12])
        jl, jt = struct.unpack('<II', h[12:20])
        if magic != GLB_MAGIC or jt != JSON_CT:
            raise SystemExit(f'{path}: not a .glb')
        self.total, self.jlen = total, jl
        self.j = json.loads(_readn(self.f, jl))
        bh = _readn(self.f, 8)
        self.blen, bt = struct.unpack('<II', bh)
        if bt != BIN_CT:
            raise SystemExit(f'{path}: no BIN chunk after the JSON')
        self.bin0 = 28 + jl
        size = os.path.getsize(path)
        if size != self.bin0 + self.blen or total != size:
            raise SystemExit(f'{path}: {size} bytes, the header says {total} and the chunks {self.bin0 + self.blen}')
        bufs = self.j.get('buffers', [])
        if len(bufs) != 1 or 'uri' in bufs[0]:
            raise SystemExit(f'{path}: not one embedded buffer')

    def where(self, ai):
        """(file offset, dtype, components, count, stride) of accessor ai"""
        a = self.j['accessors'][ai]
        v = self.j['bufferViews'][a['bufferView']]
        dt, size = _CT[a['componentType']]
        nc = _NC[a['type']]
        stride = v.get('byteStride') or size * nc
        return self.bin0 + v.get('byteOffset', 0) + a.get('byteOffset', 0), dt, nc, a['count'], stride

    def read(self, ai):
        off, dt, nc, cnt, stride = self.where(ai)
        self.f.seek(off)
        raw = _readn(self.f, stride * (cnt - 1) + np.dtype(dt).itemsize * nc) if cnt else b''
        return np.ndarray((cnt, nc), dtype=dt, buffer=raw, strides=(stride, np.dtype(dt).itemsize)).copy()


def _runs(idx):
    """sorted vertex indices -> [(first, last+1)] contiguous runs"""
    if not len(idx):
        return []
    cut = np.nonzero(np.diff(idx) != 1)[0]
    starts = np.concatenate([[idx[0]], idx[cut + 1]])
    ends = np.concatenate([idx[cut] + 1, [idx[-1] + 1]])
    return list(zip(starts.tolist(), ends.tolist()))


def _merge(J, X, base):
    """extras.glb's JSON X into the map's J, its binary at byte `base` of the
    map's buffer: only what its meshes and extras nodes use"""
    used_mats, used_acc = set(), set()
    for m in X['meshes']:
        for pr in m['primitives']:
            used_acc.update(pr['attributes'].values())
            if 'indices' in pr:
                used_acc.add(pr['indices'])
            if pr.get('material') is not None:
                used_mats.add(pr['material'])
    used_tex = set()
    for mi in used_mats:
        m = X['materials'][mi]
        for k in ('baseColorTexture', 'metallicRoughnessTexture'):
            t = (m.get('pbrMetallicRoughness') or {}).get(k)
            if t is not None:
                used_tex.add(t['index'])
        for k in ('normalTexture', 'emissiveTexture', 'occlusionTexture'):
            if m.get(k) is not None:
                used_tex.add(m[k]['index'])
    used_img = {X['textures'][t]['source'] for t in used_tex}
    used_bv = {X['accessors'][a]['bufferView'] for a in used_acc} | {X['images'][i]['bufferView'] for i in used_img}
    bvm, acm, imm, txm, mtm, smm, msm, ndm = {}, {}, {}, {}, {}, {}, {}, {}
    for i in sorted(used_bv):
        v = dict(X['bufferViews'][i])
        v['buffer'] = 0
        v['byteOffset'] = base + v.get('byteOffset', 0)
        J['bufferViews'].append(v)
        bvm[i] = len(J['bufferViews']) - 1
    for i in sorted(used_acc):
        a = dict(X['accessors'][i])
        a['bufferView'] = bvm[a['bufferView']]
        J['accessors'].append(a)
        acm[i] = len(J['accessors']) - 1
    for i in sorted(used_img):
        im = dict(X['images'][i])
        im['bufferView'] = bvm[im['bufferView']]
        J.setdefault('images', []).append(im)
        imm[i] = len(J['images']) - 1
    for i, s in enumerate(X.get('samplers', [])):
        J.setdefault('samplers', []).append(dict(s))
        smm[i] = len(J['samplers']) - 1
    for i in sorted(used_tex):
        t = dict(X['textures'][i])
        t['source'] = imm[t['source']]
        if 'sampler' in t:
            t['sampler'] = smm[t['sampler']]
        J.setdefault('textures', []).append(t)
        txm[i] = len(J['textures']) - 1

    def retex(d):
        if isinstance(d, dict):
            if 'index' in d and isinstance(d['index'], int) and set(d) <= {'index', 'texCoord', 'scale', 'strength', 'extensions'}:
                d = dict(d)
                d['index'] = txm[d['index']]
                return d
            return {k: retex(v) for k, v in d.items()}
        return d
    for i in sorted(used_mats):
        m = json.loads(json.dumps(X['materials'][i]))
        for k in ('normalTexture', 'emissiveTexture', 'occlusionTexture'):
            if k in m:
                m[k] = retex(m[k])
        if 'pbrMetallicRoughness' in m:
            m['pbrMetallicRoughness'] = {k: (retex(v) if k.endswith('Texture') else v)
                                         for k, v in m['pbrMetallicRoughness'].items()}
        J['materials'].append(m)
        mtm[i] = len(J['materials']) - 1
    for i, m in enumerate(X['meshes']):
        m = json.loads(json.dumps(m))
        for pr in m['primitives']:
            pr['attributes'] = {k: acm[v] for k, v in pr['attributes'].items()}
            if 'indices' in pr:
                pr['indices'] = acm[pr['indices']]
            if pr.get('material') is not None:
                pr['material'] = mtm[pr['material']]
        J['meshes'].append(m)
        msm[i] = len(J['meshes']) - 1
    n0 = len(J['nodes'])
    for i, n in enumerate(X['nodes']):
        n = json.loads(json.dumps(n))
        if 'mesh' in n:
            n['mesh'] = msm[n['mesh']]
        if 'children' in n:
            n['children'] = [c + n0 for c in n['children']]
        J['nodes'].append(n)
    J['scenes'][J.get('scene', 0)]['nodes'] += [r + n0 for r in X['scenes'][X.get('scene', 0)]['nodes']]
    for e in X.get('extensionsUsed', []):
        if e not in J.setdefault('extensionsUsed', []):
            J['extensionsUsed'].append(e)
    return used_bv


def _prune_empty(J):
    """leftover empty nodes (no mesh, no children, nothing in them) out: room
    in the JSON chunk"""
    drop = {i for i, n in enumerate(J['nodes'])
            if set(n) <= {'name', 'translation', 'rotation', 'scale', 'matrix'}}
    roots = set(J['scenes'][J.get('scene', 0)]['nodes'])
    drop -= roots
    keep = [i for i in range(len(J['nodes'])) if i not in drop]
    m = {o: k for k, o in enumerate(keep)}
    nodes = []
    for o in keep:
        n = J['nodes'][o]
        if 'children' in n:
            n['children'] = [m[c] for c in n['children'] if c in m]
            if not n['children']:
                del n['children']
        nodes.append(n)
    J['nodes'] = nodes
    for s in J['scenes']:
        s['nodes'] = [m[r] for r in s['nodes'] if r in m]
    return len(drop)


def cmd_patch(a):
    W = a.work
    dmeta = json.load(open(os.path.join(W, 'doors.json')))['doors'] if os.path.isfile(os.path.join(W, 'doors.json')) else []
    dz = np.load(os.path.join(W, 'doors.npz')) if dmeta else None
    ex = os.path.join(W, 'extras.glb')
    xinfo = json.load(open(os.path.join(W, 'extras.json'))) if os.path.isfile(os.path.join(W, 'extras.json')) else {}
    # the doors' vertices, keyed to the millimetre; a key two doors share is
    # nobody's
    kk, kd = [], []
    for i, m in enumerate(dmeta):
        k = np.unique(_keys(dz[f't{i}']))
        kk.append(k)
        kd.append(np.full(len(k), i, np.int64))
    if kk:
        K = np.concatenate(kk); Dd = np.concatenate(kd)
        o = np.argsort(K, kind='stable'); K, Dd = K[o], Dd[o]
        dup = np.zeros(len(K), bool)
        dup[1:] |= K[1:] == K[:-1]
        dup[:-1] |= K[1:] == K[:-1]
        Dd[dup] = -1
        K, ui = np.unique(K, return_index=True)
        Dd = Dd[ui]
    # where each moved prop's corners go, and its turn (for the normals)
    moves = {}
    for i, m in enumerate(dmeta):
        if m['action'] != 'move':
            continue
        To = dz[f't{i}'].astype(np.float64).reshape(-1, 3)
        Tn = dz[f'n{i}'].astype(np.float64).reshape(-1, 3)
        ko = _keys(To)
        uk, first = np.unique(ko, return_index=True)
        a0, b0 = To - To.mean(0), Tn - Tn.mean(0)
        U_, _S, Vt = np.linalg.svd(b0.T @ a0)
        Rm = U_ @ np.diag([1, 1, np.sign(np.linalg.det(U_ @ Vt))]) @ Vt
        moves[i] = (uk, Tn[first], Rm)
    for path in a.glb:
        _patch_one(path, dmeta, K if kk else None, Dd if kk else None, ex, xinfo, rewrite=a.rewrite,
                   moves=moves)


def _rewrite(g, js, writes, xblob):
    """the whole file again, through a temporary copy: a JSON that no longer
    fits in its chunk"""
    path = g.path
    js += b' ' * ((4 - len(js) % 4) % 4)
    delta = len(js) - g.jlen
    tmp = path + '.tmp'
    with open(tmp, 'wb') as o:
        o.write(struct.pack('<III', GLB_MAGIC, 2, 28 + len(js) + g.blen + len(xblob)))
        o.write(struct.pack('<II', len(js), JSON_CT)); o.write(js)
        o.write(struct.pack('<II', g.blen + len(xblob), BIN_CT))
        g.f.seek(g.bin0)
        left = g.blen
        while left:
            b = _readn(g.f, min(left, 64 << 20))
            o.write(b)
            left -= len(b)
        o.write(xblob)
        for off, new, old in writes:
            o.seek(off + delta); o.write(new)
    g.f.close()
    os.replace(tmp, path)
    GlbFile(path).f.close()
    print(f'   done, written again whole: {path} ({os.path.getsize(path) / 1e6:.1f} MB) - no undo for this one')


def _patch_one(path, dmeta, K, Dd, ex, xinfo, rewrite=False, moves=None):
    g = GlbFile(path)
    J = g.j
    if (J.get('asset', {}).get('extras') or {}).get('zone_extras'):
        print(f'{path}: already patched ({J["asset"]["extras"]["zone_extras"]}) - `undo` first to do it again')
        g.f.close()
        return
    undo = path + '.extras_undo'
    if os.path.exists(undo):
        raise SystemExit(f'{undo} is there from a run that did not finish: `undo` first')
    # ---- the doors, in the light pages
    writes = []                       # (file offset, new bytes, old bytes)
    found = collections.Counter()
    bounds = {}
    if K is not None:
        for ni, n in enumerate(J['nodes']):
            if 'mesh' not in n or not PAGE_RE.match(n.get('name', '')):
                continue
            for pr in J['meshes'][n['mesh']]['primitives']:
                at = pr['attributes']
                P = g.read(at['POSITION'])
                k = _keys(P)
                pos = np.searchsorted(K, k)
                pos = np.clip(pos, 0, len(K) - 1)
                hit = K[pos] == k
                who = np.where(hit, Dd[pos], -1)
                if 'indices' in pr:
                    I = g.read(pr['indices']).reshape(-1, 3).astype(np.int64)
                else:
                    I = np.arange(len(P)).reshape(-1, 3)
                w3 = who[I]
                ok = (w3[:, 0] >= 0) & (w3[:, 0] == w3[:, 1]) & (w3[:, 0] == w3[:, 2])
                if not ok.any():
                    continue
                newP = P.astype(np.float64).copy()
                N = g.read(at['NORMAL']).astype(np.float64) if 'NORMAL' in at else None
                newN = N.copy() if N is not None else None
                changed = np.zeros(len(P), bool)
                for di in np.unique(w3[ok, 0]):
                    tsel = I[ok & (w3[:, 0] == di)]
                    m = dmeta[di]
                    found[di] += len(tsel)
                    vs = np.unique(tsel.reshape(-1))
                    if m['action'] == 'move':
                        mk, mp, mr = moves[di]
                        kv = k[vs]
                        j = np.clip(np.searchsorted(mk, kv), 0, len(mk) - 1)
                        okv = mk[j] == kv
                        newP[vs[okv]] = mp[j[okv]]
                        if newN is not None:
                            newN[vs] = N[vs] @ mr.T
                    elif m['action'] == 'turn':
                        D = np.array(m['D'])
                        newP[vs] = P[vs].astype(np.float64) @ D[:3, :3].T + D[:3, 3]
                        if newN is not None:
                            newN[vs] = N[vs] @ D[:3, :3].T
                    else:
                        # left out: each triangle folded to a point
                        c = P.astype(np.float64)[tsel].mean(axis=1)
                        for col in range(3):
                            newP[tsel[:, col]] = c
                    changed[vs] = True
                idx = np.nonzero(changed)[0]
                for acc, arr in ((at['POSITION'], newP), (at.get('NORMAL'), newN)):
                    if acc is None or arr is None:
                        continue
                    off, dt, nc, cnt, stride = g.where(acc)
                    if stride != 12 or dt != '<f4':
                        raise SystemExit('interleaved or non-float vertex data: not handled')
                    for s, e in _runs(idx):
                        g.f.seek(off + 12 * s)
                        old = _readn(g.f, 12 * (e - s))
                        writes.append((off + 12 * s, arr[s:e].astype('<f4').tobytes(), old))
                pa = J['accessors'][at['POSITION']]
                pa['min'] = [float(x) for x in newP.astype(np.float32).min(axis=0)]
                pa['max'] = [float(x) for x in newP.astype(np.float32).max(axis=0)]
    turned = sum(1 for i in found if dmeta[i]['action'] in ('turn', 'move'))
    dropped = sum(1 for i in found if dmeta[i]['action'] == 'drop')
    short = [f"{dmeta[i]['name']} ({found.get(i, 0)}/{dmeta[i]['triangles']})" for i in range(len(dmeta))
             if found.get(i, 0) < dmeta[i]['triangles']]
    print(f'{os.path.basename(os.path.dirname(path))}/map.glb: {turned} of {sum(1 for m in dmeta if m["action"] in ("turn", "move"))} props '
          f'moved (doors opened, poses put right), {dropped} of {sum(1 for m in dmeta if m["action"] == "drop")} animated props folded away, '
          f'{sum(found.values()):,} triangles, {sum(len(w[1]) for w in writes) / 1e6:.2f} MB of vertices')
    if short:
        print(f'   not every triangle found for {len(short)}: {", ".join(short[:12])}{" ..." if len(short) > 12 else ""}')
    # ---- the extras
    xblob = b''
    if os.path.isfile(ex):
        xg = GlbFile(ex)
        xg.f.seek(xg.bin0)
        xblob = _readn(xg.f, xg.blen)
        X = xg.j
        xg.f.close()
        base = g.blen                       # a multiple of 4: the chunk is padded
        used_bv = _merge(J, X, base)
        # only what is used of its binary: still one block, laid out as it was
        end = max(X['bufferViews'][i].get('byteOffset', 0) + X['bufferViews'][i]['byteLength'] for i in used_bv)
        xblob = xblob[:end]
        xblob += b'\0' * ((4 - len(xblob) % 4) % 4)
        if xinfo.get('sky_colour'):
            SKY_c = [float(min(max(c, 0.0), 1.0)) for c in xinfo['sky_colour']]
            for mm in J['materials']:
                if mm.get('name') == SKY.NAME:
                    mm.setdefault('pbrMetallicRoughness', {})['baseColorFactor'] = SKY_c + [1.0]
                    mm['emissiveFactor'] = SKY_c
        print(f'   + extras.glb: {len(X["nodes"])} nodes, {len(used_bv)} buffer views ({len(xblob) / 1e6:.1f} MB)')
    J['buffers'][0]['byteLength'] = g.blen + len(xblob)
    J.setdefault('asset', {}).setdefault('extras', {})['zone_extras'] = {
        'tool': 'add_extras', 'props_moved': turned, 'props_left_out': dropped,
        'teleports': xinfo.get('teleports', 0), 'sky3d': bool(xinfo.get('sky3d'))}
    js = json.dumps(J, separators=(',', ':')).encode('utf-8')
    pruned = 0
    if len(js) > g.jlen:
        pruned = _prune_empty(J)
        js = json.dumps(J, separators=(',', ':')).encode('utf-8')
    if len(js) > g.jlen:
        if not rewrite:
            raise SystemExit(f'the new JSON is {len(js) - g.jlen} bytes over the old chunk: --rewrite '
                             f'writes the whole file again (slow for a big map on a slow disk)')
        _rewrite(g, js, writes, xblob)
        return
    js += b' ' * (g.jlen - len(js))
    if pruned:
        print(f'   {pruned} leftover empty nodes taken out, to fit the JSON in its place')
    # ---- the originals first, then the new bytes
    g.f.seek(0)
    head = _readn(g.f, 28 + g.jlen)
    with open(undo, 'wb') as u:
        u.write(struct.pack('<QQ', os.path.getsize(path), len(writes)))
        u.write(struct.pack('<Q', len(head))); u.write(head)
        for off, new, old in writes:
            u.write(struct.pack('<QQ', off, len(old))); u.write(old)
    g.f.close()
    with open(path, 'r+b') as f:
        f.seek(0, 2)
        f.write(xblob)
        for off, new, old in writes:
            f.seek(off); f.write(new)
        f.seek(20 + g.jlen); f.write(struct.pack('<II', g.blen + len(xblob), BIN_CT))
        f.seek(20); f.write(js)
        f.seek(0); f.write(struct.pack('<III', GLB_MAGIC, 2, 28 + g.jlen + g.blen + len(xblob)))
    chk = GlbFile(path)               # reads back as a whole .glb, every view inside it
    over = [i for i, v in enumerate(chk.j['bufferViews']) if v.get('byteOffset', 0) + v['byteLength'] > chk.blen]
    chk.f.close()
    if over:
        raise SystemExit(f'{path}: {len(over)} buffer views past the end - `undo` it')
    print(f'   done: {path} ({os.path.getsize(path) / 1e6:.1f} MB); the originals in {os.path.basename(undo)}')


def cmd_undo(a):
    for path in a.glb:
        undo = path + '.extras_undo'
        with open(undo, 'rb') as u:
            size, nw = struct.unpack('<QQ', _readn(u, 16))
            hl = struct.unpack('<Q', _readn(u, 8))[0]
            head = _readn(u, hl)
            regions = []
            for _ in range(nw):
                off, ln = struct.unpack('<QQ', _readn(u, 16))
                regions.append((off, _readn(u, ln)))
        with open(path, 'r+b') as f:
            for off, old in regions:
                f.seek(off); f.write(old)
            f.seek(0); f.write(head)
            f.truncate(size)
        GlbFile(path).f.close()
        os.replace(undo, undo + '.used')
        print(f'{path}: back as it was ({size / 1e6:.1f} MB)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for c in ('build', 'doors'):
        p = sub.add_parser(c)
        p.add_argument('map'); p.add_argument('report'); p.add_argument('work')
        p.add_argument('--game-dir')
    p = sub.add_parser('patch'); p.add_argument('work'); p.add_argument('glb', nargs='+')
    p.add_argument('--rewrite', action='store_true',
                   help='when the JSON no longer fits in place, write the whole file again')
    p = sub.add_parser('undo'); p.add_argument('glb', nargs='+')
    a = ap.parse_args()
    {'build': cmd_build, 'doors': cmd_doors, 'patch': cmd_patch, 'undo': cmd_undo}[a.cmd](a)


if __name__ == '__main__':
    main()
