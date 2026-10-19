"""What a Source map does beyond its geometry, carried into the .glb for the
map_extras mod (zone-fpv-mods): its teleports, its 3D skybox, and the props
its own logic opens.

Teleports. Many maps join their areas with trigger_teleport brushes - walk
into a closet, come out in another level - and some areas can be reached no
other way (gm_br_pitfalls has 40: a hub and its levels). Each becomes an empty
node named zone_teleport_<n> whose glTF extras hold, in the map's own space:
the trigger's corners (the mod makes an Area3D of their hull), the
destination and the way it faces, the landmark if it has one (Source then keeps
the offset from it, so the move is seamless), and whether the map starts it
disabled or switches it with its logic. Empty nodes have no mesh, so the game
neither draws nor collides with them, and without the mod they do nothing.

3D skybox. A Source map's sky is drawn through its sky brushes as a separate
scene: the 3D skybox, a 1/16-scale room around sky_camera, seen from
sky_camera + (eye / 16). gm_br_pitfalls' neighbourhood gets its night from it
- a black room with a starfield and skyscrapers - while its 2D sky is day.
With sky walls on, the room is kept where it is, as nodes named
zone_3dsky_<...>, and an empty node zone_3dsky holds what the mod needs to
draw it on the sky surfaces from the right place: where sky_camera is and
where the map's origin is, in the map's space, and the scale.

Props the map opens. prop_door_rotating doors are opened (--doors open) as
the map's brush doors are: turned about their hinge by their `distance`, the
way `opendir` allows - all but the ones it keeps locked for good (fake doors
on a wall: locked, and nothing ever unlocks them). A prop_dynamic the map's logic animates open
(SetAnimation ...open...) - a blast door, a hatch, a cabinet - cannot be
animated here, so it is left out instead, which opens the way it closed.
"""
import collections, math
import numpy as np
from . import build as BU

TELEPORT = 'zone_teleport'
SKY3D = 'zone_3dsky'


def is_sky3d_node(node):
    """a part of the 3D skybox room (not the empty zone_3dsky anchor)"""
    return str(node.get('name') or '').startswith(SKY3D + '_')


def outputs(bsp):
    """every output in the map: [(source entity, output, target, input, param)]"""
    out = []
    for e in bsp.entities:
        for k, v in e.items():
            if not k.startswith('On'):
                continue
            for line in str(v).split('\n'):
                p = line.split('\x1b') if '\x1b' in line else line.split(',')
                if len(p) >= 2:
                    out.append((e, k, p[0].strip().lower(), p[1].strip().lower(),
                                p[2].strip().lower() if len(p) > 2 else ''))
    return out


def animated_open(bsp):
    """targetnames (lower case) of the props the map's logic animates open"""
    return {t for _, _, t, inp, par in outputs(bsp)
            if inp == 'setanimation' and 'open' in par and 'close' not in par}


LOCKED = 2048                # prop_door_rotating: "Starts locked"
OPENERS = ('unlock', 'open', 'toggle', 'openawayfrom', 'setposition')


def door_logic(bsp):
    """targetnames (lower case) the map's logic unlocks or opens"""
    return {t for _, _, t, inp, _ in outputs(bsp) if inp in OPENERS}


def door_stays_shut(ent, logic):
    """a prop door the map keeps shut for good: locked from the start, not
    starting open, and nothing in the map ever unlocks or opens it - a fake
    door on a wall (gm_br_pitfalls has 43: opened, they stood out of the
    wall, and the backs of one-sided ones were invisible but solid)"""
    try:
        flags = int(float(ent.get('spawnflags', 0) or 0))
    except ValueError:
        flags = 0
    if not flags & LOCKED or str(ent.get('spawnpos', '0')).strip() in ('1', '2'):
        return False
    return str(ent.get('targetname', '')).lower() not in logic


def door_yaw(ent):
    """how far to turn a prop_door_rotating to open it, in degrees of yaw"""
    try:
        dist = float(ent.get('distance', 90) or 90)
    except ValueError:
        dist = 90.0
    pos = str(ent.get('spawnpos', '0')).strip()
    if pos == '1':
        return dist              # already starts open: forward
    if pos == '2':
        return -dist
    opendir = str(ent.get('opendir', '0')).strip()
    return -dist if opendir == '2' else dist


def teleports(bsp, to_map, dir_to_map):
    """the map's trigger_teleports, in the map's space"""
    names = {}
    for e in bsp.entities:
        if e.get('targetname'):
            names.setdefault(e['targetname'].lower(), e)
    switched = {t for _, _, t, inp, _ in outputs(bsp) if inp in ('enable', 'disable', 'toggle')}
    out = []
    for e in bsp.entities:
        if e.get('classname') != 'trigger_teleport':
            continue
        dest = names.get(str(e.get('target', '')).lower())
        mdl = str(e.get('model', ''))
        if dest is None or not mdl.startswith('*'):
            continue
        mi = int(mdl[1:])
        R, t = BU.brush_entity_transform(bsp, mi, e, doors='closed')
        m = bsp.models[mi]
        pts = []
        for fi in range(int(m['firstface']), int(m['firstface']) + int(m['numfaces'])):
            loop = bsp.face_loop(fi)
            if len(loop):
                pts.append(bsp.verts[loop].astype(np.float64))
        if pts:
            P = np.concatenate(pts) @ R.T + t
        else:          # no faces kept: its box
            lo, hi = np.array(m['mins'], np.float64), np.array(m['maxs'], np.float64)
            P = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) + t
        P = np.unique(np.round(P, 2), axis=0)
        planes = _hull_planes(bsp, m, R, t, P)
        do = BU._vec(dest.get('origin', '0 0 0'))
        da = BU._vec(dest.get('angles', '0 0 0'))
        fwd = BU._angles_to_forward(0.0, da[1], 0.0)
        lm = names.get(str(e.get('landmark', '')).lower()) if e.get('landmark') else None
        tn = str(e.get('targetname', ''))
        out.append({
            'name': tn, 'target': str(e.get('target', '')),
            'points': [[round(float(c), 3) for c in p] for p in to_map(P)],
            # inside is n.x <= d for every one: the mod tests the drone's path
            # against them, so a fast drone cannot pass through between ticks
            'planes': _planes_to_map(planes, to_map, dir_to_map),
            'dest': [round(float(c), 3) for c in to_map(do[None, :])[0]],
            'dest_forward': [round(float(c), 5) for c in dir_to_map(fwd)],
            'landmark': ([round(float(c), 3) for c in to_map(BU._vec(lm.get('origin', '0 0 0'))[None, :])[0]]
                         if lm is not None else None),
            'start_disabled': str(e.get('StartDisabled', '0')).strip() == '1',
            'switched': tn.lower() in switched if tn else False,
        })
    return out


def add_teleports(glb, tps):
    """one empty node each, its trigger in the extras -> [node index]"""
    return [_node_with_extras(glb, f'{TELEPORT}_{i}', {TELEPORT: tp}) for i, tp in enumerate(tps)]


def _node_with_extras(glb, name, extras):
    i = glb.node(name)
    glb.j['nodes'][i]['extras'] = extras
    return i


def add_sky3d(glb, anchor, origin, scale):
    """the empty node that tells the mod how to draw the 3D skybox"""
    return _node_with_extras(glb, SKY3D, {SKY3D: {
        'anchor': [round(float(c), 4) for c in anchor],
        'origin': [round(float(c), 4) for c in origin],
        'scale': float(scale),
        # how the room's light is held (light_sky3d): COLOR_0 = sRGB of half
        'light': LIGHT_ENCODING}})


def _hull_planes(bsp, m, R, t, P, eps=0.5):
    """[(n, d)] in Hammer units: the trigger's faces whose plane has all of it
    behind - for the usual one-brush trigger, the brush - and its box"""
    out = []
    cen = P.mean(axis=0)
    for fi in range(int(m['firstface']), int(m['firstface']) + int(m['numfaces'])):
        loop = bsp.face_loop(fi)
        if len(loop) < 3:
            continue
        q = bsp.verts[loop].astype(np.float64) @ R.T + t
        n = np.zeros(3)
        for k in range(len(q)):                          # Newell
            a, b = q[k], q[(k + 1) % len(q)]
            n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]),
                           (a[0] - b[0]) * (a[1] + b[1])])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        if n @ (q.mean(axis=0) - cen) < 0:
            n = -n
        d = float(n @ q[0])
        if (P @ n - d).max() <= eps:
            out.append((n, d))
    lo, hi = P.min(axis=0), P.max(axis=0)
    for ax in range(3):
        e = np.zeros(3); e[ax] = 1.0
        out += [(e, float(hi[ax])), (-e, float(-lo[ax]))]
    keep = []
    for n, d in out:
        if not any(n @ k[0] > 0.9999 and abs(d - k[1]) < 0.01 for k in keep):
            keep.append((n, d))
    return keep


def _planes_to_map(planes, to_map, dir_to_map):
    res = []
    for n, d in planes:
        nm = np.asarray(dir_to_map(n), np.float64)
        nm /= max(np.linalg.norm(nm), 1e-12)
        pm = to_map((n * d)[None, :])[0]                 # a point on the plane
        res.append([round(float(c), 5) for c in nm] + [round(float(nm @ pm), 4)])
    return res


# ---------------------------------------------------------------- 3D skybox
# The room is never baked (bake_glb leaves it out): it is drawn by the mod
# from its own camera, so it carries its light with it - Valve's own, the
# lightmaps vrad baked for it, read at its vertices (zonemap/valvelight.py)
# into COLOR_0, its materials unshaded copies. A colour is 8 bits a channel
# in Godot, so it is stored as sRGB of half the light (light 1 = the albedo
# as it is, Source's overbright reaching 2): the mod decodes it.
SURF_NOLIGHT = 0x400
LIGHT_FULL = 128.0
LIGHT_ENCODING = 'srgb_half'
SUBDIV_UNITS = 24.0          # Hammer units: the finest light detail carried
SKY3D_TRI_BUDGET = 60000     # cut no finer than this many triangles in all


def unlit_materials(bsp):
    """lower-case names of the materials vrad gave no lightmap (SURF_NOLIGHT:
    starfields, skies, light panels) - shown as they are"""
    out = set()
    for fi in range(len(bsp.faces)):
        name, flags, _w, _h = bsp.face_material(fi)
        if name is not None and flags & SURF_NOLIGHT:
            out.add(name.lower())
    return out


def _srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def encode_light(L):
    """Valve light (lightmap units) -> COLOR_0"""
    return _srgb(np.clip(np.asarray(L, np.float64) / LIGHT_FULL, 0.0, 2.0) / 2.0).astype(np.float32)


_CT = {5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}
_NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def _read(glb, ai):
    a = glb.j['accessors'][ai]
    v = glb.j['bufferViews'][a['bufferView']]
    dt = np.dtype(_CT[a['componentType']])
    nc = _NC[a['type']]
    off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = v.get('byteStride') or dt.itemsize * nc
    arr = np.ndarray((a['count'], nc), dtype=dt, buffer=glb.blob, offset=off,
                     strides=(stride, dt.itemsize))
    return np.array(arr)


def _subdivide(P, N, U, max_len):
    """(T,3,3) triangles -> finer ones, no edge longer than max_len, with their
    normals and uvs interpolated; winding kept"""
    edge = np.stack([np.linalg.norm(P[:, 1] - P[:, 0], axis=1),
                     np.linalg.norm(P[:, 2] - P[:, 1], axis=1),
                     np.linalg.norm(P[:, 0] - P[:, 2], axis=1)], 1).max(1)
    nseg = np.clip(np.ceil(edge / max_len), 1, 64).astype(np.int64)
    outP, outN, outU = [], [], []
    for n in np.unique(nseg):
        sel = np.nonzero(nseg == n)[0]
        pts, ix = {}, []

        def vid(i, j):
            return pts.setdefault((i, j), len(pts))
        for i in range(n):
            for j in range(n - i):
                ix.append((vid(i, j), vid(i + 1, j), vid(i, j + 1)))
                if i + j < n - 1:
                    ix.append((vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)))
        W = np.zeros((len(pts), 3))
        for (i, j), k in pts.items():
            W[k] = (1.0 - (i + j) / n, i / n, j / n)
        T = W[np.array(ix).reshape(-1)]                  # (3t, 3) weights
        outP.append(np.einsum('vk,tkc->tvc', T, P[sel]).reshape(-1, 3, 3))
        outN.append(np.einsum('vk,tkc->tvc', T, N[sel]).reshape(-1, 3, 3))
        outU.append(np.einsum('vk,tkc->tvc', T, U[sel]).reshape(-1, 3, 2))
    return np.concatenate(outP), np.concatenate(outN), np.concatenate(outU)


def _unlit_copy(glb, mi, cache):
    if mi in cache:
        return cache[mi]
    import copy
    m = copy.deepcopy(glb.j['materials'][mi])
    m['name'] = f"{SKY3D}_{m.get('name', mi)}"
    for k in ('normalTexture', 'occlusionTexture', 'emissiveTexture', 'emissiveFactor'):
        m.pop(k, None)
    m.setdefault('extensions', {})['KHR_materials_unlit'] = {}
    used = glb.j.setdefault('extensionsUsed', [])
    if 'KHR_materials_unlit' not in used:
        used.append('KHR_materials_unlit')
    glb.j['materials'].append(m)
    cache[mi] = len(glb.j['materials']) - 1
    return cache[mi]


def _albedo_mean(glb, mi, cache):
    """a material's mean albedo, linear"""
    if mi in cache:
        return cache[mi]
    m = glb.j['materials'][mi] if mi is not None else {}
    pbr = m.get('pbrMetallicRoughness', {})
    c = np.array(pbr.get('baseColorFactor', [1, 1, 1, 1])[:3], np.float64)
    t = pbr.get('baseColorTexture')
    if t is not None:
        try:
            import io
            from PIL import Image
            img = glb.j['images'][glb.j['textures'][t['index']]['source']]
            v = glb.j['bufferViews'][img['bufferView']]
            o = v.get('byteOffset', 0)
            im = Image.open(io.BytesIO(bytes(glb.blob[o:o + v['byteLength']]))).convert('RGB')
            im.thumbnail((64, 64))
            a = np.asarray(im, np.float64).reshape(-1, 3) / 255.0
            lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
            c = c * lin.mean(axis=0)
        except Exception:
            c = c * 0.5
    cache[mi] = c
    return c


def light_sky3d(glb, bsp, chunk_nodes, prop_nodes, to_src, dir_to_src, anchor,
                unlit=(), sky_tris=None, sky_colour=(0.56, 0.69, 0.86),
                max_units=SUBDIV_UNITS, budget=SKY3D_TRI_BUDGET, verbose=True):
    """Valve's light on the 3D skybox room, which no bake reaches.

    chunk_nodes: [(node, material key)] - the room's brushwork, its triangles
    cut finer (no edge over max_units) and given the light at each corner;
    prop_nodes: [node] - its props, each the light where it stands. All get
    unshaded copies of their materials. Returns (stats, the colour the room
    shows from sky_camera on average - the sky surfaces' fallback - or None).
    """
    from .valvelight import LightField
    J = glb.j
    st = {'triangles_in': 0, 'triangles': 0, 'exact': 0, 'inexact': 0, 'props': 0}
    pts = []
    for ni, _m in chunk_nodes:
        for pr in J['meshes'][J['nodes'][ni]['mesh']]['primitives']:
            pts.append(_read(glb, pr['attributes']['POSITION']))
    for ni in prop_nodes:
        pts.append(np.array([J['nodes'][ni].get('translation', [0, 0, 0])], np.float64))
    if not pts:
        return st, None
    src = to_src(np.concatenate(pts))
    lo, hi = src.min(axis=0) - 64.0, src.max(axis=0) + 64.0
    field = LightField(bsp, verbose=False,
                       keep=lambda c: bool((c >= lo).all() and (c <= hi).all()))
    scale_m = np.linalg.norm(to_src(np.array([[0.0, 0, 0], [1.0, 0, 0]]))[1]
                             - to_src(np.array([[0.0, 0, 0]]))[0])     # units a metre
    # as fine as max_units, unless that is more triangles than the budget
    # (a desert of terrain round a small map's skybox): then coarser
    lit_edges = []
    for ni, mkey in chunk_nodes:
        if str(mkey).split('#')[0] in unlit:
            continue
        for pr in J['meshes'][J['nodes'][ni]['mesh']]['primitives']:
            P = _read(glb, pr['attributes']['POSITION']).astype(np.float64)
            I = (_read(glb, pr['indices']).reshape(-1) if 'indices' in pr else np.arange(len(P)))
            q = P[I.reshape(-1, 3)]
            lit_edges.append(np.stack([np.linalg.norm(q[:, k] - q[:, (k + 1) % 3], axis=1)
                                       for k in range(3)], 1).max(1))
    max_m = max_units / max(scale_m, 1e-9)
    if lit_edges:
        e = np.concatenate(lit_edges)
        while (np.clip(np.ceil(e / max_m), 1, 64) ** 2).sum() > budget and max_m < 1e4:
            max_m *= 1.25
    st['cut_to_units'] = round(float(max_m * scale_m), 1)
    cache, amean = {}, {}
    seen_tris, seen_cols = [], []
    anchor = np.asarray(anchor, np.float64)

    def seen(P, C):
        # kept for what the room shows from sky_camera (_view_colour)
        q = np.asarray(P, np.float64).reshape(-1, 3, 3)
        C = np.asarray(C, np.float64)
        if len(C) == 3 * len(q):
            C = C.reshape(-1, 3, 3).mean(axis=1)
        seen_tris.append(q)
        seen_cols.append(C)

    for ni, mkey in chunk_nodes:
        node = J['nodes'][ni]
        mesh = J['meshes'][node['mesh']]
        new_prims = []
        for pr in mesh['primitives']:
            at = pr['attributes']
            P = _read(glb, at['POSITION']).astype(np.float64)
            N = (_read(glb, at['NORMAL']).astype(np.float64) if 'NORMAL' in at
                 else np.zeros_like(P))
            U = (_read(glb, at['TEXCOORD_0']).astype(np.float64) if 'TEXCOORD_0' in at
                 else np.zeros((len(P), 2)))
            I = (_read(glb, pr['indices']).reshape(-1) if 'indices' in pr
                 else np.arange(len(P)))
            tri = I.reshape(-1, 3)
            st['triangles_in'] += len(tri)
            bare = str(mkey).split('#')[0]
            if bare in unlit:
                Pt, Nt, Ut = P[tri], N[tri], U[tri]
                L = np.full((Pt.shape[0] * 3, 3), LIGHT_FULL)
            else:
                Pt, Nt, Ut = _subdivide(P[tri], N[tri], U[tri], max_m)
                L = np.zeros((Pt.shape[0] * 3, 3))
                S = to_src(Pt.reshape(-1, 3)).reshape(-1, 3, 3)
                fn = np.cross(S[:, 1] - S[:, 0], S[:, 2] - S[:, 0])
                fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
                dd = (fn * S[:, 0]).sum(1)
                key = np.concatenate([np.round(fn * 50), np.round(dd / 2.0)[:, None]], 1)
                _u, inv = np.unique(key, axis=0, return_inverse=True)
                for g in range(len(_u)):
                    ts = np.nonzero(inv.reshape(-1) == g)[0]
                    vp = S[ts].reshape(-1, 3)
                    best = None
                    for sgn in (1.0, -1.0):          # a two-sided copy faces back
                        s1 = collections.Counter()
                        li = field.sample(vp, sgn * fn[ts[0]], s1)
                        if best is None or s1['exact'] > best[1]['exact']:
                            best = (li, s1)
                        if s1['exact'] == len(vp):
                            break
                    L[(ts[:, None] * 3 + np.arange(3)).reshape(-1)] = best[0]
                    st['exact'] += best[1]['exact']
                    st['inexact'] += best[1]['inexact']
            col = encode_light(L)
            mat = pr.get('material')
            nm = _unlit_copy(glb, mat, cache) if mat is not None else None
            seen(Pt.reshape(-1, 3), (np.minimum(L / LIGHT_FULL, 2.0)
                                     * _albedo_mean(glb, mat, amean)[None, :]))
            n_ = Nt.reshape(-1, 3).astype(np.float32)
            n_ /= np.maximum(np.linalg.norm(n_, axis=1, keepdims=True), 1e-9)
            new_prims.append((Pt.reshape(-1, 3).astype(np.float32), n_,
                              Ut.reshape(-1, 2).astype(np.float32), col, nm))
            st['triangles'] += Pt.shape[0]
        node['mesh'] = _mesh_with_colour(glb, mesh.get('name', 'sky3d'), new_prims)
    # props: the light where each stands, on every vertex
    for ni in prop_nodes:
        node = J['nodes'][ni]
        o = to_src(np.array([node.get('translation', [0, 0, 0])], np.float64))
        s1 = collections.Counter()
        li = field.sample(o + np.array([[0, 0, 1.0]]), np.array([0, 0, 1.0]), s1)[0]
        col = encode_light(li[None, :])[0]
        mesh = J['meshes'][node['mesh']]
        prims = []
        for pr in mesh['primitives']:
            p2 = dict(pr)
            at = dict(pr['attributes'])
            cnt = J['accessors'][at['POSITION']]['count']
            at['COLOR_0'] = glb._acc(np.tile(col, (cnt, 1)).astype(np.float32), 5126, 'VEC3', 34962)
            p2['attributes'] = at
            if pr.get('material') is not None:
                p2['material'] = _unlit_copy(glb, pr['material'], cache)
            prims.append(p2)
        J['meshes'].append({'name': mesh.get('name', 'prop') + '_sky3d', 'primitives': prims})
        node['mesh'] = len(J['meshes']) - 1
        st['props'] += 1
    if sky_tris is not None and len(sky_tris):
        seen(np.asarray(sky_tris, np.float64).reshape(-1, 3),
             np.tile(np.asarray(sky_colour, np.float64), (len(sky_tris) // 3, 1)))
    st['materials'] = len(cache)
    colour = (_view_colour(anchor, np.concatenate(seen_tris), np.concatenate(seen_cols))
              if seen_tris else None)
    if verbose:
        tot = max(st['exact'] + st['inexact'], 1)
        print(f"[4/5] 3D skybox room lit with the map's own light: {st['triangles']:,} triangles "
              f"(from {st['triangles_in']:,}, cut to {st['cut_to_units']:g} units), "
              f"{100.0 * st['exact'] / tot:.0f} % read straight off "
              f"its lightmaps, {st['props']} props, {st['materials']} unshaded materials")
    return st, colour


def _mesh_with_colour(glb, name, prims):
    out = []
    for pos, nrm, uv, col, mat in prims:
        at = {'POSITION': glb._acc(pos, 5126, 'VEC3', 34962, minmax=True),
              'NORMAL': glb._acc(nrm, 5126, 'VEC3', 34962),
              'TEXCOORD_0': glb._acc(uv, 5126, 'VEC2', 34962),
              'COLOR_0': glb._acc(col, 5126, 'VEC3', 34962)}
        p = {'attributes': at, 'mode': 4,
             'indices': glb._acc(np.arange(len(pos), dtype='<u4'), 5125, 'SCALAR', 34963)}
        if mat is not None:
            p['material'] = int(mat)
        out.append(p)
    glb.j['meshes'].append({'name': name, 'primitives': out})
    return len(glb.j['meshes']) - 1


def _view_colour(eye, tris, cols, n=1200):
    """the average of what is seen from eye, in every direction: the colour of
    the first triangle each of n rays meets (or none), by Moller-Trumbore"""
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    th = math.pi * (1.0 + 5 ** 0.5) * k
    dirs = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    s_ = eye - v0
    q_ = np.cross(s_, e1)
    t_ = (e2 * q_).sum(1)
    acc, hits = np.zeros(3), 0
    for d in dirs:
        p = np.cross(d, e2)
        det = (e1 * p).sum(1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        u = (s_ * p).sum(1) * inv
        w = (q_ @ d) * inv
        t = t_ * inv
        hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 1e-6)
        if hit.any():
            i = np.nonzero(hit)[0][np.argmin(t[hit])]
            acc += cols[i]
            hits += 1
    return acc / hits if hits else None
