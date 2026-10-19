"""The map's sky brushes, kept as sky surfaces: zone_sky.

A Source map is closed in by sky brushes - walls and a ceiling of
tools/toolsskybox that the engine paints with the sky. They are what a player
sees as sky, and they hide everything beyond them: the map's other areas (the
levels of a backrooms map, a room built out of sight) and the outside of its
own buildings. The Zone's sky is only a backdrop at infinity, so with them
dropped, as every tool texture is, the whole map shows from the street
(gm_br_pitfalls: its other levels floating over the neighbourhood).

So their faces are kept, as nodes named zone_sky_<cell> with one material, and
are solid, as in the original. The map_extras mod (zone-fpv-mods) draws the
sky on them - the map's own 3D skybox where it has one, else the game's sky,
matched to the sky behind; without it they are a plain colour, unshaded: a
clear sky's blue, or what the 3D skybox shows on average (a night map's
dark). Faces of the miniature 3D skybox room are dropped with
it. bake_glb leaves them out of the light bake.
"""
import collections
import numpy as np
from . import bsp as B, build as BU

NAME = 'zone_sky'
COLOUR = (0.56, 0.69, 0.86)     # the fallback: a clear sky's blue, unshaded


def faces(bsp, scale=BU.UNIT):
    """The world's sky faces as triangles in glTF space (before any shift):
    -> (positions (3n, 3), normals (3n, 3)); the front faces the map"""
    mdl = bsp.models[0]
    disp = {int(d['mapface']) for d in bsp.dispinfo}
    P, N = [], []
    for fi in range(int(mdl['firstface']), int(mdl['firstface']) + int(mdl['numfaces'])):
        name, flags, _w, _h = bsp.face_material(fi)
        if name is None or not flags & (B.SURF_SKY | B.SURF_SKY2D) or fi in disp:
            continue
        loop = bsp.face_loop(fi)
        if len(loop) < 3:
            continue
        vs = bsp.verts[loop].astype(np.float64)
        nrm = np.array(bsp.planes[int(bsp.faces[fi]['planenum'])]['normal'], np.float64)
        # counter-clockwise about the plane normal, which points into the map
        # (as build.extract does for every face)
        if np.cross(vs, np.roll(vs, -1, axis=0)).sum(axis=0) @ nrm < 0:
            vs = vs[::-1].copy()
        k = len(vs)
        fan = np.stack([np.zeros(k - 2, np.int64), np.arange(1, k - 1), np.arange(2, k)], axis=1)
        P.append(BU.to_gltf(vs[fan].reshape(-1, 3), scale))
        N.append(np.tile(BU.to_gltf(nrm[None, :], 1.0), (3 * (k - 2), 1)))
    if not P:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.concatenate(P).astype(np.float32), np.concatenate(N).astype(np.float32)


def area(pos):
    q = np.asarray(pos, np.float64).reshape(-1, 3, 3)
    return float(0.5 * np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1).sum())


def keep_tris(pos, nrm, keep):
    """the triangles whose flag in keep is True"""
    m = np.repeat(np.asarray(keep, bool), 3)
    return pos[m], nrm[m]


def add(glb, pos, nrm, cell_m=48.0, colour=COLOUR):
    """The sky faces as zone_sky nodes, one per cell -> [node index]"""
    if not len(pos):
        return []
    mat = glb.material(NAME, base_color=tuple(colour) + (1.0,), roughness=1.0,
                       emissive=colour)
    glb.j['materials'][mat]['extensions'] = {'KHR_materials_unlit': {}}
    used = glb.j.setdefault('extensionsUsed', [])
    if 'KHR_materials_unlit' not in used:
        used.append('KHR_materials_unlit')
    tri = np.asarray(pos, np.float32).reshape(-1, 3, 3)
    nt = np.asarray(nrm, np.float32).reshape(-1, 3, 3)
    cells = collections.defaultdict(list)
    for t, c in enumerate(np.floor(tri.mean(axis=1) / cell_m).astype(np.int64)):
        cells[tuple(int(x) for x in c)].append(t)
    nodes = []
    for c, ts in sorted(cells.items()):
        p = tri[ts].reshape(-1, 3)
        n = nt[ts].reshape(-1, 3)
        idx = np.arange(len(p), dtype=np.uint32)
        nm = f'{NAME}_{c[0]}_{c[1]}_{c[2]}'
        nodes.append(glb.node(nm, mesh=glb.mesh(nm, [(p, n, None, idx, mat)])))
    return nodes


def recolour(glb, colour):
    """the sky surfaces' plain colour (the fallback, without the mod)"""
    c = [float(min(max(x, 0.0), 1.0)) for x in colour]
    for m in glb.j['materials']:
        if m.get('name') == NAME:
            m.setdefault('pbrMetallicRoughness', {})['baseColorFactor'] = c + [1.0]
            m['emissiveFactor'] = c


def is_sky_node(node):
    return str(node.get('name') or '').startswith(NAME)
