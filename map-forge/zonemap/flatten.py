"""The map's turn, put into the map itself.

forge.py --recenter turns the whole map about the spawn so the drone starts
facing the game's default, -Z (manifest.spawn_turn). That turn used to sit on
the glTF's root node: right in the game, but everything that read the map had
to undo it (bake_glb's light lookup, the Cycles lights), and in Blender the
map arrived as an empty turned 90 degrees with everything crooked under it.

flatten_roots() moves a root node's transform down into the map:

  * a world mesh - a node with no transform of its own, no children, and the
    only user of its mesh - gets it in its vertices (positions, normals,
    tangents) and keeps no transform;
  * anything placed - props, doors, a mesh used more than once, groups - gets
    it in its own translation and rotation, so a prop's rotation is its facing
    in the map as you fly it;
  * the root node keeps its name and children, and no transform.

Every vertex stays where it was in the world (to float precision), so the game
shows the map exactly as before. It works on the writer's Glb (j, blob) and on
a loaded .glb's JSON and binary chunk, as long as that is a bytearray.
"""
import math
import numpy as np

_XFORM = ('matrix', 'translation', 'rotation', 'scale')
_F32 = 5126


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def q_to_m(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def m_to_q(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s,
             (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s,
             (R[1, 0] - R[0, 1]) / s]
    n = math.sqrt(sum(c * c for c in q)) or 1.0
    return [float(c / n) for c in q]


def node_matrix(n):
    if 'matrix' in n:
        return np.array(n['matrix'], np.float64).reshape(4, 4).T
    M = np.eye(4)
    M[:3, :3] = q_to_m(n.get('rotation', [0, 0, 0, 1])) @ np.diag(n.get('scale', [1, 1, 1]))
    M[:3, 3] = n.get('translation', [0, 0, 0])
    return M


def is_identity(n):
    return np.allclose(node_matrix(n), np.eye(4), atol=1e-7)


def rigid(M):
    """M -> (t, q, s) when it is a turn, a move and one uniform scale; else None"""
    L = M[:3, :3]
    s = float(np.cbrt(np.linalg.det(L)))
    if s <= 1e-9 or not np.allclose(L.T @ L, np.eye(3) * s * s, atol=1e-5 * max(1.0, s * s)):
        return None
    return M[:3, 3].astype(np.float64), m_to_q(L / s), s


def yaw_deg(q):
    """the turn about +Y a quaternion is, in degrees; None if it tips the map"""
    x, y, z, w = q
    if abs(x) > 1e-6 or abs(z) > 1e-6:
        return None
    return (math.degrees(2.0 * math.atan2(y, w)) + 180.0) % 360.0 - 180.0


def root_turns(j):
    """[(node index, its matrix)] for each scene root that carries a transform"""
    sc = j.get('scenes', [{}])[j.get('scene', 0)]
    return [(r, node_matrix(j['nodes'][r])) for r in sc.get('nodes', [])
            if any(k in j['nodes'][r] for k in _XFORM) and not is_identity(j['nodes'][r])]


def _set_trs(n, t, q, s):
    for k in _XFORM:
        n.pop(k, None)
    if not np.allclose(t, 0.0, atol=1e-7):
        n['translation'] = [float(c) for c in t]
    if not np.allclose(q, [0, 0, 0, 1], atol=1e-9):
        n['rotation'] = [float(c) for c in q]
    if not np.allclose(s, 1.0, atol=1e-9):
        n['scale'] = [float(c) for c in s]


def _floats(j, blob, ai):
    """a writable (count, n) float32 view of accessor ai, or None"""
    a = j['accessors'][ai]
    n = {'VEC3': 3, 'VEC4': 4}.get(a.get('type'))
    if (n is None or a.get('componentType') != _F32 or 'sparse' in a
            or a.get('bufferView') is None):
        return None
    v = j['bufferViews'][a['bufferView']]
    bufs = j.get('buffers') or []
    if v.get('buffer', 0) != 0 or (bufs and 'uri' in bufs[0]):
        return None
    off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = v.get('byteStride') or 4 * n
    return np.ndarray((a['count'], n), dtype='<f4', buffer=blob, offset=off,
                      strides=(stride, 4))


def flatten_roots(j, blob):
    """Move every scene root's transform into the map (see the module notes).
    -> stats dict, or None when no root carries a transform"""
    turns = root_turns(j)
    if not turns:
        return None
    nodes, meshes = j['nodes'], j.get('meshes', [])
    mesh_users = np.zeros(len(meshes), int)
    for n in nodes:
        if 'mesh' in n:
            mesh_users[n['mesh']] += 1
    acc_users = {}
    for mi, m in enumerate(meshes):
        for p in m.get('primitives', []):
            refs = list(p.get('attributes', {}).values())
            for tg in p.get('targets', []):
                refs += list(tg.values())
            for ai in set(refs):
                acc_users.setdefault(ai, set()).add(mi)
    st = {'roots': 0, 'turn_deg': None, 'baked_meshes': 0, 'baked_vertices': 0,
          'placed_nodes': 0, 'refused': []}
    done_acc = set()
    for r, M in turns:
        root = nodes[r]
        trs = rigid(M)
        if trs is None:
            st['refused'].append(f"{root.get('name', r)}: its transform is not a "
                                 f"turn, a move and a uniform scale")
            continue
        t0, q0, s0 = trs
        R0 = q_to_m(q0)
        # anything hung on the root itself goes on a child of its own first
        own = {k: root.pop(k) for k in ('mesh', 'camera') if k in root}
        lt = root.get('extensions', {}).pop('KHR_lights_punctual', None)
        if lt is not None:
            own['extensions'] = {'KHR_lights_punctual': lt}
            if not root['extensions']:
                root.pop('extensions')
        if own:
            nodes.append(dict(name=f"{root.get('name', 'root')}_own", **own))
            root.setdefault('children', []).append(len(nodes) - 1)
        for c in root.get('children', []):
            n = nodes[c]
            mi = n.get('mesh')
            plain = not any(k in n for k in _XFORM) or is_identity(n)
            if (plain and mi is not None and not n.get('children') and 'skin' not in n
                    and mesh_users[mi] == 1):
                prims = meshes[mi].get('primitives', [])
                views, ok = [], all(not p.get('targets') for p in prims)
                for p in prims if ok else ():
                    for key in ('POSITION', 'NORMAL', 'TANGENT'):
                        ai = p.get('attributes', {}).get(key)
                        if ai is None or ai in done_acc or any(ai == w[1] for w in views):
                            continue
                        arr = _floats(j, blob, ai)
                        if arr is None or acc_users.get(ai, set()) - {mi}:
                            ok = False
                            break
                        views.append((key, ai, arr))
                    if not ok:
                        break
                if ok:
                    for key, ai, arr in views:
                        if key == 'POSITION':
                            arr[:] = (arr.astype(np.float64) @ (R0 * s0).T + t0).astype('<f4')
                            a = j['accessors'][ai]
                            if len(arr):
                                a['min'] = [float(x) for x in arr.min(axis=0)]
                                a['max'] = [float(x) for x in arr.max(axis=0)]
                            st['baked_vertices'] += len(arr)
                        else:
                            arr[:, :3] = (arr[:, :3].astype(np.float64) @ R0.T).astype('<f4')
                        done_acc.add(ai)
                    for k in _XFORM:
                        n.pop(k, None)
                    st['baked_meshes'] += 1
                    continue
            # placed: the turn goes into its own transform
            if 'matrix' in n:
                n['matrix'] = [float(x) for x in (M @ node_matrix(n)).T.reshape(-1)]
            else:
                tc = np.array(n.get('translation', [0, 0, 0]), np.float64)
                qc = n.get('rotation', [0, 0, 0, 1])
                sc = np.array(n.get('scale', [1, 1, 1]), np.float64)
                q = q_mul(q0, qc)
                qn = math.sqrt(sum(x * x for x in q)) or 1.0
                _set_trs(n, t0 + s0 * (R0 @ tc), [x / qn for x in q], s0 * sc)
            st['placed_nodes'] += 1
        for k in _XFORM:
            root.pop(k, None)
        st['roots'] += 1
        st['turn_deg'] = yaw_deg(q0) if st['turn_deg'] is None else st['turn_deg']
    return st


def describe(st):
    if not st:
        return 'no turned root node'
    turn = (f"{st['turn_deg']:+.0f} deg" if st['turn_deg'] is not None else 'its transform')
    s = (f"turned {turn} in the map itself: {st['baked_meshes']} world meshes in their "
         f"vertices, {st['placed_nodes']} placed nodes in their own transforms; "
         f"the root node is left unturned")
    if st['refused']:
        s += '\n      NOT moved: ' + '; '.join(st['refused'])
    return s
