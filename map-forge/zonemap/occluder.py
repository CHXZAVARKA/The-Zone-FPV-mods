"""occluder.bin: what the game culls the map against.

The game (custommaps/handler/custom_map_root.gd, load_occluder) reads a map
folder's occluder.bin with gdutil/z_occluder_bin.gd - "ZOCC", u32 version 1,
u32 vertex count, u32 index count, the vertices as float32 x, y, z, the
indices as int32 - into an ArrayOccluder3D, and Godot then skips every node
whose bounds are hidden behind it (occlusion culling is on in the game's
project and in its 3D view). bake_glb used to write it empty.

The occluders are the map's big, solid surfaces: world geometry (not props),
opaque materials only (no cut-outs, glass or water), triangles of at least
min_area, and the sky surfaces (zone_sky), which are what hide one area of a
map from the next. At most max_tris, the biggest first.
"""
import struct
import numpy as np

MIN_AREA = 0.25          # m2
MAX_TRIS = 300000


def _matrix(n):
    if 'matrix' in n:
        return np.array(n['matrix'], np.float64).reshape(4, 4).T
    x, y, z, w = n.get('rotation', [0, 0, 0, 1])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R @ np.diag(n.get('scale', [1, 1, 1]))
    M[:3, 3] = n.get('translation', [0, 0, 0])
    return M


def is_prop(name):
    nm = str(name or '').lower()
    return 'prop_' in nm or nm.startswith('prop')


def from_glb(j, acc, min_area=MIN_AREA, max_tris=MAX_TRIS, skip=None):
    """The occluder triangles of a map .glb: j its JSON, acc(i) an accessor's
    array -> (vertices (n, 3) float32, indices (m,) int32), and what was kept"""
    world = {}

    def walk(i, M):
        Mi = M @ _matrix(j['nodes'][i])
        world[i] = Mi
        for c in j['nodes'][i].get('children', ()):
            walk(c, Mi)
    for r in j['scenes'][j.get('scene', 0)]['nodes']:
        walk(r, np.eye(4))
    tris, kinds = [], {'world': 0, 'sky': 0}
    for ni, n in enumerate(j['nodes']):
        if 'mesh' not in n or ni not in world:
            continue
        name = n.get('name', '')
        sky = str(name).startswith('zone_sky')
        if str(name).startswith('zone_3dsky'):
            continue        # the 3D skybox room, drawn from its own camera
        if not sky and (is_prop(name) or (skip and skip(name))):
            continue
        M = world[ni]
        for pr in j['meshes'][n['mesh']]['primitives']:
            m = j['materials'][pr['material']] if pr.get('material') is not None else {}
            if not sky and m.get('alphaMode', 'OPAQUE') != 'OPAQUE':
                continue
            at = pr.get('attributes', {})
            if 'POSITION' not in at or pr.get('mode', 4) != 4:
                continue
            P = np.asarray(acc(at['POSITION']), np.float64)
            P = P @ M[:3, :3].T + M[:3, 3]
            idx = (np.asarray(acc(pr['indices']), np.int64).reshape(-1, 3) if 'indices' in pr
                   else np.arange(len(P)).reshape(-1, 3))
            q = P[idx]
            ar = 0.5 * np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
            keep = ar >= min_area
            if keep.any():
                tris.append((q[keep].astype(np.float32), ar[keep]))
                kinds['sky' if sky else 'world'] += int(keep.sum())
    if not tris:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32), kinds
    Q = np.concatenate([t for t, _ in tris])
    A = np.concatenate([a for _, a in tris])
    if len(Q) > max_tris:
        Q = Q[np.argsort(-A, kind='stable')[:max_tris]]
    # one vertex per position (to the millimetre)
    flat = Q.reshape(-1, 3)
    key = np.round(flat * 1000.0).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return flat[first].astype(np.float32), inv.reshape(-1).astype(np.int32), dict(kinds, kept=len(Q))


def write(path, verts, idx):
    verts = np.ascontiguousarray(verts, '<f4').reshape(-1, 3)
    idx = np.ascontiguousarray(idx, '<i4').reshape(-1)
    with open(path, 'wb') as f:
        f.write(b'ZOCC' + struct.pack('<III', 1, len(verts), len(idx)))
        f.write(verts.tobytes())
        f.write(idx.tobytes())
    return 16 + verts.nbytes + idx.nbytes
