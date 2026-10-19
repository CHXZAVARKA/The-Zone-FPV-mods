"""The pose a model is drawn in: its first sequence's first frame.

A .vvd holds a model's vertices as they were compiled; the game draws them
through the bones, each vertex moved by boneToWorld * poseToBone of the bones
it hangs on. poseToBone undoes the bind pose, and boneToWorld is the pose the
model is drawn in - the frame of the animation it plays, not the bind pose.
Where the two agree (nearly every model, and every $staticprop, whose bones are
collapsed) the vertices are drawn where they are; where they do not, the
model is drawn turned or moved.

That is common for doors and switches: the default sequence ("idle") turns
the root bone a quarter turn from how the mesh was built. gm_br_pitfalls has 43
such models - 22 office doors, its "pitfalls doors", the HL2 door01_left,
light switches, levers, a cargo elevator - and without this every one of them
stood a quarter turn off: doors sticking straight out of their walls (the one
README section on .vvd vertices was right that applying the BIND pose is
wrong: it is the animation's pose, which for those inferno models is the bind
pose again, so they stay put).

Read here: sequence 0, its first blend, frame 0 - raw or RLE-compressed
rotations and positions (Quaternion48/64, Vector48, mstudioanimvalue_t),
local to the .mdl or in its .ani animation block, with bones the animation
leaves out at their default. Delta (additive) animations are left alone.
"""
import math, struct
import numpy as np

ANIM_RAWPOS, ANIM_RAWROT, ANIM_ANIMPOS, ANIM_ANIMROT, ANIM_DELTA, ANIM_RAWROT2 = 1, 2, 4, 8, 16, 32
BONE_SIZE = 216


def _half(h):
    return float(np.frombuffer(struct.pack('<H', h), dtype='<f2')[0])


def _q48(b, o):
    x, y, zw = struct.unpack_from('<3H', b, o)
    qx = (x - 32768) / 32768.0
    qy = (y - 32768) / 32768.0
    qz = ((zw & 0x7FFF) - 16384) / 16384.0
    w = math.sqrt(max(0.0, 1.0 - qx * qx - qy * qy - qz * qz))
    return (qx, qy, qz, -w if zw & 0x8000 else w)


def _q64(b, o):
    v, = struct.unpack_from('<Q', b, o)
    x = (v & 0x1FFFFF); y = (v >> 21) & 0x1FFFFF; z = (v >> 42) & 0x1FFFFF; wneg = v >> 63
    qx = (x - 1048576) / 1048576.5
    qy = (y - 1048576) / 1048576.5
    qz = (z - 1048576) / 1048576.5
    w = math.sqrt(max(0.0, 1.0 - qx * qx - qy * qy - qz * qz))
    return (qx, qy, qz, -w if wneg else w)


def _euler_quat(a):
    """Source's AngleQuaternion(RadianEuler)"""
    sy, cy = math.sin(a[2] * 0.5), math.cos(a[2] * 0.5)
    sp, cp = math.sin(a[1] * 0.5), math.cos(a[1] * 0.5)
    sr, cr = math.sin(a[0] * 0.5), math.cos(a[0] * 0.5)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def _quat_mat(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _value0(b, o):
    """frame 0 of an RLE mstudioanimvalue_t run at o"""
    valid, total = struct.unpack_from('<BB', b, o)
    if total == 0:
        return 0.0
    v, = struct.unpack_from('<h', b, o + 2) if valid > 0 else (0,)
    return float(v)


def frame0(mdl_blob, read=None, seq=0):
    """bone -> model matrices drawn at sequence `seq`, frame 0:
    (R (n,3,3), T (n,3), poseToBone (n,3,4)) or None (with why)"""
    b = mdl_blob
    nbones, boneidx = struct.unpack_from('<2i', b, 156)
    nanim, animidx = struct.unpack_from('<2i', b, 180)
    nseq, seqidx = struct.unpack_from('<2i', b, 188)
    if nbones <= 0 or nseq <= 0 or nanim <= 0 or seq >= nseq:
        return None, 'no sequence'
    bones = []
    for i in range(nbones):
        o = boneidx + i * BONE_SIZE
        parent, = struct.unpack_from('<i', b, o + 4)
        pos = struct.unpack_from('<3f', b, o + 32)
        rot = struct.unpack_from('<3f', b, o + 60)
        posscale = struct.unpack_from('<3f', b, o + 72)
        rotscale = struct.unpack_from('<3f', b, o + 84)
        p2b = np.array(struct.unpack_from('<12f', b, o + 96), np.float64).reshape(3, 4)
        bones.append((parent, pos, rot, posscale, rotscale, p2b))
    so = seqidx + seq * 212
    animindexindex, = struct.unpack_from('<i', b, so + 60)
    ai, = struct.unpack_from('<h', b, so + animindexindex)
    if ai < 0 or ai >= nanim:
        return None, 'bad anim index'
    ao = animidx + ai * 100
    aflags, = struct.unpack_from('<i', b, ao + 12)
    if aflags & 0x0004:                       # STUDIO_DELTA
        return None, 'delta'
    animblock, animindex = struct.unpack_from('<2i', b, ao + 52)
    sectionindex, sectionframes = struct.unpack_from('<2i', b, ao + 80)
    if sectionframes:
        animblock, animindex = struct.unpack_from('<2i', b, ao + sectionindex)
        base = ao
    src, start = b, ao + animindex
    if animblock:
        # in the .ani file: studiohdr numanimblocks/animblockindex at 352
        nblk, blkidx = struct.unpack_from('<2i', b, 352)
        if read is None or animblock >= nblk:
            return None, 'animation block not read'
        ds, _de = struct.unpack_from('<2i', b, blkidx + animblock * 8)
        ani = read()
        if ani is None:
            return None, '.ani missing'
        src, start = ani, ds + animindex
    local = {}
    o = start
    for _ in range(nbones + 1):
        if o + 4 > len(src):
            break
        bone, flags, nxt = struct.unpack_from('<BBh', src, o)
        if bone >= nbones:
            break
        d = o + 4
        if flags & ANIM_DELTA:
            return None, 'delta'
        bb = bones[bone]
        q = None
        if flags & ANIM_RAWROT2:
            q = _q64(src, d); posat = d + 8
        elif flags & ANIM_RAWROT:
            q = _q48(src, d); posat = d + 6
        elif flags & ANIM_ANIMROT:
            offs = struct.unpack_from('<3h', src, d)
            e = [bb[2][k] + (_value0(src, d + offs[k]) * bb[4][k] if offs[k] else 0.0) for k in range(3)]
            q = _euler_quat(e); posat = d + 6
        else:
            posat = d
        p = None
        if flags & ANIM_RAWPOS:
            hx, hy, hz = struct.unpack_from('<3H', src, posat)
            p = (_half(hx), _half(hy), _half(hz))
        elif flags & ANIM_ANIMPOS:
            pv = d + (6 if flags & ANIM_ANIMROT else 0)
            offs = struct.unpack_from('<3h', src, pv)
            p = tuple(bb[1][k] + (_value0(src, pv + offs[k]) * bb[3][k] if offs[k] else 0.0) for k in range(3))
        local[bone] = (q, p)
        if nxt == 0:
            break
        o += nxt
    R = np.zeros((nbones, 3, 3)); T = np.zeros((nbones, 3)); P2B = np.zeros((nbones, 3, 4))
    for i, (parent, pos, rot, _ps, _rs, p2b) in enumerate(bones):
        q, p = local.get(i, (None, None))
        lr = _quat_mat(q if q is not None else _euler_quat(rot))
        lt = np.array(p if p is not None else pos, np.float64)
        if 0 <= parent < i:
            R[i] = R[parent] @ lr
            T[i] = R[parent] @ lt + T[parent]
        else:
            R[i], T[i] = lr, lt
        P2B[i] = p2b
    return (R, T, P2B), None


def skin(vvd, pose):
    """the .vvd's positions and normals, drawn in this pose -> (pos, nrm, moved)"""
    R, T, P2B = pose
    n = len(R)
    # per bone: M = boneToWorld * poseToBone (3x4)
    M = np.zeros((n, 3, 4))
    for i in range(n):
        M[i, :, :3] = R[i] @ P2B[i, :, :3]
        M[i, :, 3] = R[i] @ P2B[i, :, 3] + T[i]
    pos = np.asarray(vvd.pos, np.float64)
    nrm = np.asarray(vvd.nrm, np.float64)
    w = np.asarray(vvd.weight, np.float64)
    bi = np.clip(np.asarray(vvd.bone, np.int64), 0, n - 1)
    nb = np.clip(np.asarray(vvd.nbones, np.int64), 1, 3)
    out_p = np.zeros_like(pos); out_n = np.zeros_like(nrm)
    wsum = np.zeros(len(pos))
    for k in range(3):
        use = nb > k
        wk = np.where(use, w[:, k], 0.0)
        Mk = M[bi[:, k]]
        out_p += wk[:, None] * (np.einsum('nij,nj->ni', Mk[:, :, :3], pos) + Mk[:, :, 3])
        out_n += wk[:, None] * np.einsum('nij,nj->ni', Mk[:, :, :3], nrm)
        wsum += wk
    ok = wsum > 1e-6
    out_p[ok] /= wsum[ok, None]
    out_p[~ok] = pos[~ok]
    out_n[~ok] = nrm[~ok]
    out_n /= np.maximum(np.linalg.norm(out_n, axis=1, keepdims=True), 1e-9)
    moved = float(np.abs(out_p - pos).max()) if len(pos) else 0.0
    return out_p, out_n, moved
