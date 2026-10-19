#!/usr/bin/env python3
"""Bake a map's original lighting into a .glb that has since been edited.

`forge.py --bake-lightmaps` works off BSP faces. The moment geometry only
exists in the .glb - a roof closed by hand in Blender, a pillar modelled from
scratch - that path is gone: there is no BSP face to read luxels from, and the
.bsp on disk no longer matches what you fly.

This bakes the .glb itself:

  * every world mesh is cut into planar charts and given its own patch of a
    shared atlas;
  * each texel samples the mesh's OWN material texture through its OWN UVs, so
    hand-unwrapped geometry bakes exactly as it currently looks;
  * the light comes from the original .bsp, looked up in world space;
  * props and any mesh used by more than one node are left alone - they are
    instanced, and baking would make every copy unique.

    python tools/bake_glb.py edited.glb --bsp maps/de_cache.bsp -o baked.glb

The .glb is metres, Y-up, recentred and turned to face the spawn down -Z; the
.bsp is Hammer units, Z-up, at its own origin. `convert_report.json` (written
beside every build) carries the `scale_m_per_unit`, `origin_shift` and
`map_turn_deg` that relate them, and is picked up automatically from beside
the input .glb. Without it the shift is estimated
from the two bounding boxes - always check the reported `exact` percentage,
which is how much of the map found a real BSP face underneath. It should be
high; a few per cent means the alignment is wrong.

With --zone-v3 it writes the 2026 beta's real-lightmap map folder instead
(lighting v3: the official maps' camera): the map keeps its own tiled textures,
the light goes into HDR lightmaps (BC6H), and tools/pack_zone_map.gd (Godot 4.4+)
packs and installs it:

    python tools/bake_glb.py map.glb --bsp map.bsp --zone-v3 --install
    python tools/bake_glb.py map.glb --bsp map.bsp --zone-v3 --cycles --install

The .glb must be built WITHOUT --bake-lightmaps (no light in its textures).

Light for geometry the BSP knows nothing about comes, in order: from a
same-facing face on the same plane (an extended roof matches its neighbours
seam for seam), else from the nearest same-facing lit surface, else from a
coarse grid of the map's own luxels, so a new pillar in a dim room comes out
dim and one in a lit atrium comes out bright.
"""
import argparse, collections, json, math, os, re, struct, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zonemap import bsp as B
from zonemap.glb import Glb
from zonemap import flatten as FL
from zonemap import sky as SKY
from zonemap import extras as EX
from zonemap import pagelayout as PL
from zonemap import occluder as OCC
from zonemap.materials import _srgb_to_lin, _lin_to_srgb, _png, _jpg
from zonemap.lightmap import (_decode, EXPOSURE_PCT, LIGHT_FLOOR,
                              LIGHT_FULL, BRIGHTNESS, KNEE_AT, KNEE_FLAT,
                              tonemap, ZONE_FLAT_NORMAL, ZONE_FLAT_GAIN,
                              ZONE_NOSPEC_NORMAL, ZONE_NOSPEC_GAIN,
                              ZONE_LOW_NORMAL, ZONE_LOW_GAIN,
                              ZONE_MID_NORMAL, ZONE_MID_GAIN,
                              ZONE_BAKED_ROUGHNESS, ZONE_BEST_SKY_ROTATION,
                              ZONE_NOSPEC_GAMMA, ZONE_ADAPT_GAMMA,
                              SHADOW_PEDESTAL, ADAPT_TARGET, ADAPT_STRENGTH,
                              ADAPT_MAX, ADAPT_RADIUS_M, Adapt, shadow_floor)

PAD = 2
TEXEL_MIN_M, TEXEL_MAX_M, TEXEL_K = 0.03, 0.30, 48.0
MAT_PREFIX = '#lm'

CT = {5120: 'i1', 5121: 'u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


# ------------------------------------------------------------------ glTF in
class GlbIn:
    def __init__(self, path):
        f = open(path, 'rb')
        assert f.read(4) == b'glTF', f'not a .glb: {path}'
        struct.unpack('<II', f.read(8))
        self.j, self.bin = None, b''
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            ln, ty = struct.unpack('<II', h)
            d = f.read(ln)
            if ty == 0x4E4F534A:
                self.j = json.loads(d)
            else:
                self.bin = d
        f.close()

    def acc(self, i):
        a = self.j['accessors'][i]
        v = self.j['bufferViews'][a['bufferView']]
        off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        n = NC[a['type']]
        arr = np.frombuffer(self.bin, CT[a['componentType']],
                            count=a['count'] * n, offset=off)
        arr = arr.reshape(a['count'], n) if n > 1 else arr
        if a.get('normalized') and arr.dtype != np.float32:
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        return arr

    def view_bytes(self, i):
        v = self.j['bufferViews'][i]
        o = v.get('byteOffset', 0)
        return self.bin[o:o + v['byteLength']]

    def image_bytes(self, i):
        im = self.j['images'][i]
        if 'bufferView' in im:
            return self.view_bytes(im['bufferView']), im.get('mimeType', 'image/png')
        raise ValueError('external image URIs are not supported')

    def mat_albedo_image(self, mi):
        if mi is None or mi < 0:
            return None
        pbr = self.j['materials'][mi].get('pbrMetallicRoughness', {})
        t = pbr.get('baseColorTexture')
        if t is None:
            return None
        tex = self.j['textures'][t['index']]
        return tex.get('source')


MAX_SURFACES = 256        # Godot's RS::MAX_MESH_SURFACES: a mesh's surfaces past it are dropped
NOSOLID = 'zone_nosolid'   # zonemap.build.NOSOLID: drawn, never collided with
NS_PRIMS = set()           # (mesh, prim) of never-solid nodes, kept apart on their pages


def FL_wrap(d):
    return (float(d) + 180.0) % 360.0 - 180.0


def node_matrix(n):
    if 'matrix' in n:
        return np.array(n['matrix'], np.float64).reshape(4, 4).T
    M = np.eye(4)
    if 'scale' in n:
        M[:3, :3] = M[:3, :3] @ np.diag(n['scale'])
    if 'rotation' in n:
        x, y, z, w = n['rotation']
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M[:3, :3] = R @ M[:3, :3]
    if 'translation' in n:
        M[:3, 3] = n['translation']
    return M


def decompose(M):
    t = M[:3, 3].tolist()
    A = M[:3, :3]
    s = [float(np.linalg.norm(A[:, i])) or 1.0 for i in range(3)]
    R = A / np.array(s)[None, :]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        w = math.sqrt(1 + tr) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            d = math.sqrt(max(1e-12, 1 + R[0, 0] - R[1, 1] - R[2, 2])) * 2
            w, x, y, z = (R[2, 1] - R[1, 2]) / d, d / 4, (R[0, 1] + R[1, 0]) / d, (R[0, 2] + R[2, 0]) / d
        elif i == 1:
            d = math.sqrt(max(1e-12, 1 + R[1, 1] - R[0, 0] - R[2, 2])) * 2
            w, x, y, z = (R[0, 2] - R[2, 0]) / d, (R[0, 1] + R[1, 0]) / d, d / 4, (R[1, 2] + R[2, 1]) / d
        else:
            d = math.sqrt(max(1e-12, 1 + R[2, 2] - R[0, 0] - R[1, 1])) * 2
            w, x, y, z = (R[1, 0] - R[0, 1]) / d, (R[0, 2] + R[2, 0]) / d, (R[1, 2] + R[2, 1]) / d, d / 4
    return t, [float(x), float(y), float(z), float(w)], s


# ------------------------------------------------------------------- light
from zonemap.valvelight import LightField, GRID_UNITS, DISP_CELL, DISP_PAD  # noqa: E402,F401


# ------------------------------------------------------------------ charts
def plane_basis(n):
    a = np.zeros(3)
    a[int(np.argmin(np.abs(n)))] = 1.0
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


class Chart:
    __slots__ = ('tris', 'n', 'org', 'u', 'v', 's0', 't0', 'tw', 'th',
                 'texel', 'page', 'x', 'y', 'prim', 'lit')


def build_charts(P, tris, dot_tol=0.94, slab=0.25):
    """Grow triangles into near-planar charts.

    Not a hash of (normal, distance): bucketing splits a floor wherever it
    crosses a bucket boundary, and de_aztec came out as 28,397 two-triangle
    charts. Not strict coplanarity either - displacement terrain is subdivided
    per blend band, so no two triangles are exactly coplanar and every one
    would be its own tile.

    So: region growing. A chart starts at one triangle and swallows its
    neighbours while they stay within `dot_tol` of the chart's running average
    normal AND within `slab` metres of its plane. The angle bound keeps the
    projection from stretching (20 degrees is 6 % at the edge); the thickness
    bound stops a staircase merging into one enormous sparse tile.
    """
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    cr = np.cross(b - a, c - a)
    ln = np.linalg.norm(cr, axis=1)
    ok = ln > 1e-12
    nn = cr / np.where(ln[:, None] == 0, 1, ln[:, None])
    cen = (a + b + c) / 3.0

    at = collections.defaultdict(list)
    for i in np.nonzero(ok)[0]:
        for vi in tris[i]:
            at[tuple(np.round(P[vi] * 1000).astype(np.int64))].append(int(i))
    nb = collections.defaultdict(set)
    for _, group in at.items():
        for x in group:
            nb[x].update(group)

    done = np.zeros(len(tris), bool)
    done[~ok] = True
    order = np.argsort(-ln)
    out = []
    for seed in order:
        seed = int(seed)
        if done[seed]:
            continue
        nsum = nn[seed] * ln[seed]
        org = cen[seed]
        members = [seed]
        done[seed] = True
        queue = list(nb[seed])
        while queue:
            i = queue.pop()
            if done[i]:
                continue
            navg = nsum / np.linalg.norm(nsum)
            if nn[i] @ navg < dot_tol:
                continue
            if abs(float((cen[i] - org) @ navg)) > slab:
                continue
            done[i] = True
            members.append(i)
            nsum = nsum + nn[i] * ln[i]
            queue.extend(nb[i] - set(members))
        ch = Chart()
        ch.tris = np.array(members)
        ch.n = nsum / np.linalg.norm(nsum)
        out.append(ch)
    return out


def shelf_pack(charts, page_px, is_alpha=None):
    """Pack charts into pages, opaque first, alpha-tested onto pages of their
    own. A page is one texture and one material, so a cut-out leaf cannot share
    a page with a wall: the wall pages stay JPEG, and only the pages that need
    a cut-out pay for RGBA PNG."""
    st = {'page': 0, 'x': 0, 'y': 0, 'shelf': 0}
    used, apages = set(), set()
    if is_alpha is None:
        is_alpha = [False] * len(charts)
    for want in (False, True):
        idxs = [i for i in range(len(charts)) if bool(is_alpha[i]) is want]
        if not idxs:
            continue
        if want and used:
            st['page'] += 1
            st['x'] = st['y'] = st['shelf'] = 0
        for i in sorted(idxs, key=lambda i: -charts[i].th):
            ch = charts[i]
            bw, bh = ch.tw + 2 * PAD, ch.th + 2 * PAD
            if st['x'] + bw > page_px:
                st['x'] = 0
                st['y'] += st['shelf']
                st['shelf'] = 0
            if st['y'] + bh > page_px:
                st['page'] += 1
                st['x'] = st['y'] = st['shelf'] = 0
            ch.page, ch.x, ch.y = st['page'], st['x'] + PAD, st['y'] + PAD
            st['x'] += bw
            st['shelf'] = max(st['shelf'], bh)
            used.add(ch.page)
            if want:
                apages.add(ch.page)
    return sorted(used), sorted(apages)


def mips_of(img, keep_alpha=False):
    mode = 'RGBA' if keep_alpha else 'RGB'
    out = [np.asarray(img.convert(mode), np.float32)]
    cur = img.convert(mode)
    while min(cur.size) > 1:
        cur = cur.resize((max(1, cur.width // 2), max(1, cur.height // 2)), Image.BOX)
        out.append(np.asarray(cur, np.float32))
    return out


def sample_tex(tex, u, v):
    h, w = tex.shape[:2]
    x = np.mod((u * w).astype(np.int64), w)
    y = np.mod((v * h).astype(np.int64), h)
    return tex[y, x]


def dilate(rgb, mask, rounds=3):
    """Grow the covered texels outward, so a chart edge never samples black."""
    out = rgb.copy()
    m = mask.copy()
    for _ in range(rounds):
        if m.all():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(m.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s = np.roll(np.roll(out, dy, 0), dx, 1)
            w = np.roll(np.roll(m, dy, 0), dx, 1).astype(np.float32)
            acc += s * w[:, :, None]
            cnt += w
        new = (~m) & (cnt > 0)
        out[new] = acc[new] / cnt[new][:, None]
        m = m | new
    return out


def load_bsp(spec, mapname=None):
    if spec.lower().endswith('.gma'):
        from zonemap.gma import Gma
        arc = Gma(spec)
        maps = arc.maps()
        if not maps:
            raise SystemExit(f'no .bsp inside {spec}')
        pick = maps[0][0]
        if mapname:
            want = os.path.splitext(os.path.basename(mapname))[0].lower()
            pick = next((m for m, _ in maps
                         if os.path.splitext(os.path.basename(m))[0] == want), pick)
        print(f'[1/5] reading {pick} from the archive')
        return B.Bsp(pick, data=arc.read(pick))
    return B.Bsp(spec)


def main():
    global PAD
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('glb', help='the edited .glb to bake')
    ap.add_argument('--bsp', required=True, help='the original .bsp or .gma')
    ap.add_argument('--map', help='which map inside a .gma')
    ap.add_argument('--report', help='convert_report.json (default: beside the .glb)')
    ap.add_argument('-o', '--out', help='output .glb (default: <name>_baked.glb)')
    ap.add_argument('--texel', type=float, default=4.0, metavar='CM',
                    help='atlas density in cm per texel (default 4). 0 picks '
                         'it per chart from sqrt(area)/48, which suits the '
                         'face-sized tiles of the BSP bake but goes coarse '
                         'here, where a chart can be a whole wall')
    ap.add_argument('--page', type=int, default=2048)
    ap.add_argument('--pad', type=int, default=3, metavar='PX',
                    help=f'texels of gutter round each light chart, filled with its own edge '
                         f'(default 3; 2 before). The game halves the light pages at medium texture '
                         f'quality and quarters them at low, and mipmaps them with distance: a '
                         f'wider gutter keeps each chart\'s neighbours\' light off its edges '
                         f'longer, for more pages')
    ap.add_argument('--unlit-texel', type=float, default=16.0, metavar='CM',
                    help='density for large surfaces with no lightmap under '
                         'them at all - the rebuilt outer shell (default 16)')
    ap.add_argument('--unlit-area', type=float, default=20.0, metavar='M2',
                    help='how big an unlit chart must be to get that coarser '
                         'density (default 20)')
    ap.add_argument('--budget', type=float, default=128.0, metavar='M',
                    help='soft cap on atlas size in millions of texels '
                         '(default 128, about 1 MB of JPEG per 4 M). '
                         'Density is coarsened to fit; 0 = off')
    ap.add_argument('--exposure', type=float, default=None, metavar='L',
                    help=f'the light value that counts as fully lit (default '
                         f'{LIGHT_FULL:.0f}, which is what Source means by a '
                         'lightmap of 255 with its overbright of 2). Lower it '
                         'to brighten the map')
    ap.add_argument('--brightness', type=float, default=BRIGHTNESS,
                    metavar='F',
                    help='multiplier on the light before that (default 1.0). '
                         'Anchoring on a percentile instead left the baked '
                         'atlas at p50 19/255 against the source textures at '
                         '112, and the game pulled it back up until the '
                         'sunlit faces bloomed')
    ap.add_argument('--jpeg-q', type=int, default=90)
    ap.add_argument('--roughness-min', type=float,
                    default=ZONE_BAKED_ROUGHNESS, metavar='F',
                    help='floor on every material roughness in the output, '
                         'props included (default 0.9), and metallic to 0. '
                         'The game gives an imported material a GGX lobe that '
                         'its own baked maps never get, and below about 0.5 '
                         'that lobe is most of the pixel at the mirror angle - '
                         'the wet-plastic look. 0 leaves both alone')
    ap.add_argument('--cycles', action='store_true',
                    help="bake the light in Blender's Cycles instead of "
                         "reading Valve's lightmap: the map's own light "
                         'entities are rebuilt as Blender lights and the '
                         'irradiance is path-traced, which brings ambient '
                         'occlusion, soft shadows and colour bleed that a '
                         '2004 radiosity solve does not have')
    ap.add_argument('--blender', help='path to blender (searched if omitted)')
    ap.add_argument('--samples', type=int, default=128,
                    help='Cycles samples per texel (default 128)')
    ap.add_argument('--bounces', type=int, default=4)
    ap.add_argument('--point-scale', type=float, default=1.0, metavar='F',
                    help='multiplier on point/spot lights against the sun')
    ap.add_argument('--sun-scale', type=float, default=1.0, metavar='F')
    ap.add_argument('--no-spatial', action='store_true',
                    help='--zone-v3: pack the light pages by size alone, as before, '
                         'instead of by where the charts are (a page then spans the '
                         'whole map, so the game can cull none of it)')
    ap.add_argument('--no-occluders', action='store_true',
                    help='leave props out of the Cycles scene. Much smaller '
                         'and much faster, and every shadow they cast is lost')
    ap.add_argument('--flat-normals', nargs='?', const='auto', default=None,
                    metavar='X,Y,Z',
                    help='point every baked surface the same way so the baked '
                         'light is what renders, instead of the game lighting '
                         'the map a second time. Default "auto" is the one '
                         'direction the game responds to flatly: it faces '
                         'away from the only shadow-casting sun, so the '
                         "realtime shadow drops out and cannot leak through a "
                         'roof. Presets, dimmest and cleanest first: nospec '
                         '(no specular at all, white tops out at 123), low '
                         '(145, a faint 16/255 lobe), mid (160, 28/255), auto '
                         '(217, the 108/255 blob). Or give x,y,z yourself')
    ap.add_argument('--normal-blend', type=float, default=0.0, metavar='T',
                    help='mix T of each face\'s real normal back into the flat '
                         'one (0 = fully flat, 1 = untouched). A shared normal '
                         'makes every specular lobe line up into one map-wide '
                         'blob; any T > 0 aims each surface differently. Costs '
                         'evenness: T=0.25 swings the game\'s multiplier 1.4x '
                         'across orientations, T=0.5 swings it 5x, T=1 swings '
                         'it 100x because a ceiling gets no sky at all')
    ap.add_argument('--flat-gain', type=float, default=None, metavar='F',
                    help=f'gain on the finished albedo x light, to undo the '
                         f'constant the game multiplies a flat-normal surface '
                         f'by. Defaults to {ZONE_FLAT_GAIN} (1/0.45) with '
                         f'--flat-normals and 1.0 without. This is NOT '
                         f'--brightness: that one scales the light before the '
                         f'tone curve and so lifts the shadows, this one '
                         f'scales the finished pixel and so lifts the whole '
                         f'map, clipping the brightest albedo toward white')
    ap.add_argument('--force', action='store_true',
                    help='bake even if the input is already a baked build')
    ap.add_argument('--install', nargs='?', const='auto', metavar='DIR',
                    help="copy the result into The Zone's custom_maps as "
                         '<name>/<name>.glb with the manifest beside it. The '
                         'game loads custom_maps/<folder>/<folder>.glb, so a '
                         'file whose name does not match its folder loads as '
                         'an empty map')
    ap.add_argument('--reuse-bake', metavar='DIR',
                    help='load the light pages from an earlier --keep-scene '
                         'run instead of baking again. The bake does not '
                         'depend on --brightness or --flat-normals, so this '
                         'turns dialling those in from a four-minute loop '
                         'into a one-minute one. Only valid if nothing about '
                         'the charts changed (same --texel, --budget, --page)')
    ap.add_argument('--light-gamma', type=float, default=None, metavar='G',
                    help=f'lift the finished pixel by pow(v, G) in linear '
                         f'space. Under 1 this brightens the midtones and '
                         f'leaves the highlights where they are, which is what '
                         f'a baked map needs: the game passes a fifth of the '
                         f'light and then tonemaps with ACES, whose toe eats '
                         f'the result. Defaults to {ZONE_NOSPEC_GAMMA} with '
                         f'--flat-normals ({ZONE_ADAPT_GAMMA} when --adapt is '
                         f'on, which does most of that lifting itself), 1.0 '
                         f'without. Lower is brighter and flatter; 0.45 is '
                         f'about as far as it goes')
    ap.add_argument('--adapt', type=float, default=ADAPT_STRENGTH, metavar='S',
                    help=f'local exposure for the dark areas (Source light '
                         f'only): each point\'s light is lifted by how dark its '
                         f'neighbourhood is, as Source\'s auto exposure and an '
                         f'eye adapt, so a dim room comes up as a whole and '
                         f'keeps its own shading instead of being flattened by '
                         f'a gamma. gain = (target / level) ** S, at least 1. '
                         f'Default {ADAPT_STRENGTH}; 0 = off (the old look)')
    ap.add_argument('--adapt-max', type=float, default=ADAPT_MAX, metavar='X',
                    help=f'the most --adapt lifts a neighbourhood (default '
                         f'{ADAPT_MAX}x, two stops)')
    ap.add_argument('--adapt-target', type=float, default=ADAPT_TARGET, metavar='T',
                    help=f'the level, as a share of fully lit, that --adapt '
                         f'lifts a dark neighbourhood towards (default '
                         f'{ADAPT_TARGET}). Anywhere already brighter is left '
                         f'alone')
    ap.add_argument('--adapt-radius', type=float, default=ADAPT_RADIUS_M, metavar='M',
                    help=f'how far round a point --adapt averages the light, '
                         f'in metres (default {ADAPT_RADIUS_M})')
    ap.add_argument('--shadow-pedestal', type=float, default=SHADOW_PEDESTAL,
                    metavar='P',
                    help=f'the least light anything gets, as a share of fully '
                         f'lit (default {SHADOW_PEDESTAL}). Added under the '
                         f'light rather than clamping it, so the darkest '
                         f'corners keep their gradients - the old hard floor '
                         f'flattened the darkest eighth of de_cpl_strike')
    ap.add_argument('--sky-rotation', metavar='DEG',
                    help=f'rewrite the installed manifest\'s sky_rotation. '
                         f'It is the only skybox field the game applies to a '
                         f'custom map, and rotating the panorama rotates the '
                         f'sky ambient with it - which with a nospec normal is '
                         f'the whole of the light. "auto" uses '
                         f'{ZONE_BEST_SKY_ROTATION:.0f}, worth nearly double '
                         f'what de_cpl_strike\'s own 143 gives, for nothing '
                         f'but the visible sun no longer matching the baked '
                         f'shadows')
    ap.add_argument('--drop-material', metavar='SUBSTR',
                    help='delete every primitive whose material name contains '
                         'any of these comma-separated substrings, before '
                         'anything else looks at the map. Nothing is baked or '
                         'written for them, so the chart layout is unchanged '
                         'if they were not being baked anyway - which makes '
                         'this a two-minute A/B against --reuse-bake: drop a '
                         'suspect surface and see whether the thing you are '
                         'chasing goes with it')
    ap.add_argument('--keep-scene', action='store_true',
                    help='keep the intermediate Blender scene and light pages')
    ap.add_argument('--bake-foliage', action='store_true',
                    help='also bake alpha-tested props - trees, vines, ivy, '
                         'hay. Off by default because every leaf card is its '
                         'own chart island: on de_cpl_strike it takes the '
                         'chart count from 9k to about 60k, and the rasterise '
                         'from one minute to fifteen, for geometry that reads '
                         'as foliage catching the light either way')
    ap.add_argument('--no-bake-props', dest='bake_props', action='store_false',
                    help='leave props dynamically lit. They then render at the '
                         'game\'s own multiplier - up to 1.0 against the 0.13 '
                         'a nospec build gives the world, which is why an '
                         'unbaked crate glares next to a baked wall')
    ap.add_argument('--bake-props', dest='bake_props', action='store_true',
                    default=True,
                    help='also bake single-use prop meshes')
    ap.add_argument('--shift', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                    help="override the report's origin_shift")
    ap.add_argument('--turn', type=float, metavar='DEG',
                    help="override the report's map_turn_deg")
    ap.add_argument('--zone-v3', action='store_true',
                    help="write the beta's real-lightmap map folder (lighting "
                         "v3) instead of a .glb with the light in its textures: "
                         "textures stay sharp, light is HDR. Defaults to "
                         "--texel 10 --budget 64; --install packs it into "
                         "custom_maps/<name>_v3 with Godot")
    ap.add_argument('--godot', help='a Godot 4.4+ editor .exe for --zone-v3 '
                    '(searched in Downloads if omitted)')
    a = ap.parse_args()
    PAD = max(1, int(a.pad))
    PL.PAD = PAD
    if a.zone_v3:
        # the textures no longer ride in the atlas, so the light can be coarse:
        # Valve's own is ~33 cm a luxel on gm_br_pitfalls
        if '--texel' not in sys.argv:
            a.texel = 10.0
        if '--budget' not in sys.argv:
            a.budget = 64.0
        if '--brightness' not in sys.argv:
            # Source's "fully lit" (128) is 1.0 in the game's units, which puts
            # sunlight at 5-6; the official maps' sunlight is about 2, where the
            # shader's roll-off knee and the camera expect it
            a.brightness = 0.4
        if a.page not in (256, 512, 1024, 2048, 3072, 4096):
            raise SystemExit('--zone-v3 needs --page 256, 512, 1024, 2048, 3072 or 4096')
        a.bake_foliage = True     # leaves get lightmaps too (no albedo pages to fill)
        a.flat_normals = None
    if a.reuse_bake:
        a.cycles = True          # reusing a Cycles bake implies --cycles
    t0 = time.time()

    g = GlbIn(a.glb)
    # a map forged before the turn went into its geometry still carries it on
    # its root node: put it into the map for this bake (world positions do
    # not move), so what this writes is laid out as the game shows it
    root_turn = None
    if FL.root_turns(g.j):
        g.bin = bytearray(g.bin)
        fl = FL.flatten_roots(g.j, g.bin)
        root_turn = fl['turn_deg'] if fl and fl['turn_deg'] is not None else None
        print(f'[1/5] an older build, its turn on the root node - {FL.describe(fl)}.\n'
              f'      python tools/flatten_glb.py "{a.glb}" does that to the file, once')
    J = g.j
    if a.drop_material:
        pats = [x.strip().lower() for x in a.drop_material.split(',') if x.strip()]
        drop = {i for i, m in enumerate(J.get('materials', []))
                if any(x in (m.get('name') or '').lower() for x in pats)}
        if drop:
            gone = 0
            for m in J['meshes']:
                keep = [pr for pr in m['primitives']
                        if pr.get('material') not in drop]
                gone += len(m['primitives']) - len(keep)
                m['primitives'] = keep
            names = sorted(J['materials'][i].get('name', '?') for i in drop)
            print(f'[1/5] --drop-material: removed {gone} primitives from '
                  f'{len(drop)} material(s): {", ".join(names[:4])}'
                  f'{" ..." if len(names) > 4 else ""}')
        else:
            print(f'[1/5] --drop-material {a.drop_material!r} matched nothing')
    print(f'[1/5] {os.path.basename(a.glb)}: {len(J["meshes"])} meshes, '
          f'{len(J.get("materials", []))} materials, {len(J["nodes"])} nodes')

    # Light in, light out. A .glb built with forge.py --bake-lightmaps already
    # carries the lightmap in its albedo; baking it again multiplies the two
    # and the map comes out black in the corners and blown out in the middle.
    already = [m.get('name', '') for m in J.get('materials', [])
               if re.match(r'^(src_)?#?lm\d+$', str(m.get('name', '')))]
    if already and not a.force:
        raise SystemExit(
            f'{os.path.basename(a.glb)} is already a baked build - it has '
            f'{len(already)} atlas materials ({", ".join(already[:3])}...).\n'
            '  Baking it again applies the light twice. Convert without\n'
            '  --bake-lightmaps and bake that .glb instead:\n'
            f'    forge.py <map> --roughness-min 0.9\n'
            f'    tools/bake_glb.py <that .glb> --bsp <map.bsp> --cycles\n'
            '  (--force overrides this.)')

    # ---- how this .glb relates to the .bsp -------------------------------
    # the map is the .bsp scaled, moved (origin_shift) and turned about the
    # spawn (map_turn_deg, forge.py --recenter); the light lookups go back
    # through all three, which is all the turn costs here
    scale, shift, turn = 0.0254, np.zeros(3), root_turn or 0.0
    rep = a.report or os.path.join(os.path.dirname(os.path.abspath(a.glb)),
                                   'convert_report.json')
    if os.path.isfile(rep):
        r = json.load(open(rep))
        scale = float(r.get('scale_m_per_unit', scale))
        shift = np.array(r.get('origin_shift', [0, 0, 0]), np.float64)
        turn = float(r.get('map_turn_deg') or 0.0)
        if root_turn is not None and abs(FL_wrap(root_turn - turn)) > 0.05:
            print(f'[1/5] WARNING: the root node was turned {root_turn:+.1f} deg but '
                  f'the report says {turn:+.1f} - going by the report (--turn overrides)')
        print(f'[1/5] {os.path.basename(rep)}: scale {scale}, '
              f'origin_shift {shift.round(2).tolist()}, turn {turn:+.0f} deg')
    else:
        print(f'[1/5] no convert_report.json beside the .glb - the origin shift '
              f'will be estimated from the bounding boxes')
    if a.shift:
        shift = np.array(a.shift, np.float64)
    if a.turn is not None:
        turn = float(a.turn)
    _th = math.radians(turn)
    R = np.array([[math.cos(_th), 0.0, math.sin(_th)],
                  [0.0, 1.0, 0.0],
                  [-math.sin(_th), 0.0, math.cos(_th)]])
    a._turn_R = R

    bsp = load_bsp(a.bsp, a.map)
    light = LightField(bsp)
    adapt = None
    if not a.cycles and not a.zone_v3 and a.adapt > 0.0 and a.adapt_max > 1.0:
        adapt = Adapt(bsp, scale, target=a.adapt_target, strength=a.adapt,
                      max_gain=a.adapt_max, radius_m=a.adapt_radius)

    # ---- walk the scene --------------------------------------------------
    nodes = J['nodes']
    if a.bake_props:
        # De-instance. Two placements of the same crate stand in different
        # light, so they cannot share one atlas region. Give every node its own
        # mesh entry - the copies share accessors, so this costs a dict here
        # and duplicated vertex data only in the output (de_cpl_strike: 54,650
        # unique prop triangles become 188,495, which is nothing at 120 fps).
        seen = set()
        for nd in nodes:
            mi = nd.get('mesh')
            if mi is None:
                continue
            if mi in seen:
                J['meshes'].append(dict(J['meshes'][mi]))
                nd['mesh'] = len(J['meshes']) - 1
            else:
                seen.add(mi)
    users = collections.Counter(n['mesh'] for n in nodes if 'mesh' in n)
    wm = {}

    def walk(i, M):
        Mi = M @ node_matrix(nodes[i])
        wm[i] = Mi
        for c in nodes[i].get('children', ()):
            walk(c, Mi)

    for rt in J['scenes'][J.get('scene', 0)]['nodes']:
        walk(rt, np.eye(4))

    bake_prims = []            # (mesh, prim, world pos, nrm, uv, tris, matidx)
    skipped = collections.Counter()
    for ni, n in enumerate(nodes):
        mi = n.get('mesh')
        if mi is None:
            continue
        if (n.get('name') or '').endswith('-collider'):
            # a never-solid node's stand-in collider (convert.collider_stub):
            # nothing to light; --zone-v3 makes one per light-page node again
            skipped['collider'] += 1
            continue
        never_solid = (n.get('name') or '').startswith(NOSOLID)
        if SKY.is_sky_node(n):
            # the sky brushes (forge.py --sky walls): no light of their own -
            # the map_extras mod draws the sky on them - and passed through as
            # they are
            skipped['sky'] += 1
            continue
        if EX.is_sky3d_node(n):
            # the 3D skybox room (forge.py with sky walls): drawn by the
            # map_extras mod from its own camera, carrying the map's own
            # light (extras.light_sky3d) - passed through, never baked
            skipped['sky3d'] += 1
            continue
        if users[mi] > 1:
            skipped['instanced'] += 1
            continue
        nm = ((n.get('name') or '') + '|' + (J['meshes'][mi].get('name') or '')).lower()
        is_prop = 'prop_' in nm or nm.startswith('prop')
        if not a.bake_props and is_prop:
            skipped['prop'] += 1
            continue
        if a.bake_props and is_prop:
            skipped['prop_baked'] += 1
        M = wm.get(ni, np.eye(4))
        for pi, pr in enumerate(J['meshes'][mi]['primitives']):
            at = pr['attributes']
            if 'POSITION' not in at or 'TEXCOORD_0' not in at:
                skipped['no_uv'] += 1
                continue
            if not a.bake_foliage and is_prop:
                # Only PROP foliage is skipped. Alpha-tested WORLD geometry -
                # ivy sheets, fences, grates - is a handful of flat charts and
                # has to stay baked: de_cpl_strike hangs 1,588 m2 of
                # alpha-blended ivy canopy over its courtyards, and leaving
                # that dynamically lit puts a bright translucent veil with
                # hard brush edges across everything underneath it.
                _m = pr.get('material')
                if _m is not None and (J['materials'][_m]
                                       .get('alphaMode', 'OPAQUE') != 'OPAQUE'):
                    skipped['prop_foliage'] += 1
                    continue
            P = g.acc(at['POSITION']).astype(np.float64)
            P = P @ M[:3, :3].T + M[:3, 3]
            N = (g.acc(at['NORMAL']).astype(np.float64) @ M[:3, :3].T
                 if 'NORMAL' in at else None)
            UV = g.acc(at['TEXCOORD_0']).astype(np.float64)
            idx = (g.acc(pr['indices']).astype(np.int64).reshape(-1, 3)
                   if 'indices' in pr else
                   np.arange(len(P), dtype=np.int64).reshape(-1, 3))
            bake_prims.append([mi, pi, P, N, UV, idx, pr.get('material')])
            if never_solid:
                NS_PRIMS.add((mi, pi))
    print(f'[2/5] baking {len(bake_prims)} primitives; skipped '
          f'{dict(skipped)}')

    if not bake_prims:
        raise SystemExit('nothing to bake')

    if not os.path.isfile(rep) and not a.shift:
        allp = np.concatenate([b[2] for b in bake_prims]) @ R
        gc = (allp.min(0) + allp.max(0)) / 2
        v = bsp.verts.astype(np.float64)
        bc = (v.min(0) + v.max(0)) / 2
        bc_glb = np.array([bc[0], bc[2], -bc[1]]) * scale
        shift = gc - bc_glb
        print(f'[2/5] estimated origin_shift {shift.round(2).tolist()}')

    def to_src(P):
        P = np.asarray(P, np.float64) @ R            # world -> before the turn
        return np.stack([(P[:, 0] - shift[0]) / scale,
                         -(P[:, 2] - shift[2]) / scale,
                         (P[:, 1] - shift[1]) / scale], axis=1)

    def dir_to_src(n):
        n = np.asarray(n, np.float64) @ R
        return np.array([n[0], -n[2], n[1]], np.float64)

    # ---- charts ----------------------------------------------------------
    # Chart per MATERIAL, not per primitive. The converter chunks the world by
    # (material, 48 m cell), so one floor arrives as twenty primitives; charting
    # each on its own turns it into twenty tiles, each paying its own gutter.
    groups = collections.defaultdict(list)
    for bp in bake_prims:
        groups[bp[6]].append(bp)
    charts = []
    for mat, prims in groups.items():
        Ps, Us, Ts, own = [], [], [], []
        base = 0
        for bp in prims:
            Ps.append(bp[2]); Us.append(bp[4])
            Ts.append(bp[5] + base)
            own += [(bp[0], bp[1], k) for k in range(len(bp[5]))]
            base += len(bp[2])
        grp = {'P': np.concatenate(Ps), 'UV': np.concatenate(Us),
               'T': np.concatenate(Ts), 'own': own, 'mat': mat}
        for ch in build_charts(grp['P'], grp['T']):
            ch.prim = grp
            charts.append(ch)
    # An L-shaped or diagonal chart pays for its whole bounding box, so split
    # the wasteful ones: a long balcony rail merged into one chart can cost
    # twenty times the texels it uses.
    work, charts = charts, []
    for _ in range(5):
        nxt = []
        for ch in work:
            P = ch.prim['P']
            q3 = P[ch.prim['T'][ch.tris]]
            ch.org = q3[0, 0].copy()
            ch.u, ch.v = plane_basis(ch.n)
            s = (q3.reshape(-1, 3) - ch.org) @ ch.u
            t = (q3.reshape(-1, 3) - ch.org) @ ch.v
            ar = float(0.5 * np.linalg.norm(
                np.cross(q3[:, 1] - q3[:, 0], q3[:, 2] - q3[:, 0]), axis=1).sum())
            box = max(s.max() - s.min(), 1e-6) * max(t.max() - t.min(), 1e-6)
            if ar / box >= 0.4 or len(ch.tris) < 6:
                charts.append(ch)
                continue
            cs = s.reshape(-1, 3).mean(1)
            ct = t.reshape(-1, 3).mean(1)
            axis = cs if (s.max() - s.min()) >= (t.max() - t.min()) else ct
            cut = float(np.median(axis))
            m = axis <= cut
            if m.all() or not m.any():
                charts.append(ch)
                continue
            for sub in (m, ~m):
                c2 = Chart()
                c2.tris = ch.tris[sub]
                c2.n = ch.n
                c2.prim = ch.prim
                nxt.append(c2)
        work = nxt
        if not work:
            break
    charts += work

    for ch in charts:
        tri = ch.tris
        P = ch.prim['P']
        pts = P[ch.prim['T'][tri].reshape(-1)]
        ch.org = pts[0].copy()
        ch.u, ch.v = plane_basis(ch.n)
        s = (pts - ch.org) @ ch.u
        t = (pts - ch.org) @ ch.v
        q3 = P[ch.prim['T'][tri]]
        ar = float(0.5 * np.linalg.norm(
            np.cross(q3[:, 1] - q3[:, 0], q3[:, 2] - q3[:, 0]), axis=1).sum())
        texel = (a.texel / 100.0) if a.texel > 0 else float(
            np.clip(math.sqrt(max(ar, 1e-6)) / TEXEL_K, TEXEL_MIN_M, TEXEL_MAX_M))
        # Probe the light before sizing the tile. A big chart that finds no
        # lightmap at all is a --fill slab - the outer shell of the map, the
        # ground skirt - and its light will be one smooth grid value however
        # many texels it gets. de_aztec's world is 228,000 m2 and most of it is
        # exactly that, so giving those a coarse density is the difference
        # between a 40-page atlas and a 12-page one. Small charts keep the fine
        # density, because that is where hand-patched geometry lives.
        probe = q3[:min(6, len(q3)), 0]
        pst = collections.Counter()
        light.sample(to_src(probe), dir_to_src(ch.n), pst)
        ch.lit = pst['exact'] > 0
        if not ch.lit and ar > a.unlit_area:
            texel = max(texel, a.unlit_texel / 100.0)
        ch.texel = texel
        ch.s0, ch.t0 = s.min() - texel, t.min() - texel
        lim = a.page - 2 * PAD
        ch.tw = int(min(lim, max(2, math.ceil((s.max() - ch.s0) / texel) + 1)))
        ch.th = int(min(lim, max(2, math.ceil((t.max() - ch.t0) / texel) + 1)))
        # a chart clamped to the page limit would stretch; shrink its density
        need_w = (s.max() - ch.s0) / texel + 1
        need_h = (t.max() - ch.t0) / texel + 1
        if need_w > lim or need_h > lim:
            ch.texel = texel * max(need_w / lim, need_h / lim)
            ch.s0, ch.t0 = s.min() - ch.texel, t.min() - ch.texel
            ch.tw = int(min(lim, max(2, math.ceil((s.max() - ch.s0) / ch.texel) + 1)))
            ch.th = int(min(lim, max(2, math.ceil((t.max() - ch.t0) / ch.texel) + 1)))
    budget = a.budget * 1e6
    est = sum((c.tw + 2 * PAD) * (c.th + 2 * PAD) for c in charts)
    if budget > 0 and est > budget:
        f = math.sqrt(est / budget)
        print(f'[3/5] atlas would be {est/1e6:.0f} M texels; coarsening '
              f'density x{f:.2f} to fit --budget {a.budget}')
        lim = a.page - 2 * PAD
        for ch in charts:
            ch.texel *= f
            ch.s0 = ch.s0 + 0.0
            ch.tw = int(min(lim, max(2, math.ceil(ch.tw / f))))
            ch.th = int(min(lim, max(2, math.ceil(ch.th / f))))
    def _is_alpha(mi):
        if mi is None:
            return False
        m = J['materials'][mi]
        return m.get('alphaMode', 'OPAQUE') != 'OPAQUE'

    chart_alpha = [_is_alpha(ch.prim['mat']) for ch in charts]
    if a.zone_v3 and not a.no_spatial:
        # pages by location (zonemap/pagelayout.py): each page is one node
        # in the game, and Godot culls whole nodes
        assert PL.PAD == PAD
        cen = np.array([ch.prim['P'][ch.prim['T'][ch.tris].reshape(-1)].mean(0) for ch in charts])
        cmat = np.array([-1 if ch.prim['mat'] is None else int(ch.prim['mat']) for ch in charts])
        pg_, x_, y_, npg = PL.layout(cen, [c.tw for c in charts], [c.th for c in charts], cmat, a.page)
        for ch, p1, x1, y1 in zip(charts, pg_, x_, y_):
            ch.page, ch.x, ch.y = int(p1), int(x1), int(y1)
        pages, alpha_pages = list(range(npg)), []
    else:
        pages, alpha_pages = shelf_pack(charts, a.page, chart_alpha)
    alpha_pages = set(alpha_pages)
    texels = sum(c.tw * c.th for c in charts)
    print(f'[3/5] {len(charts)} charts -> {len(pages)} x {a.page}px pages '
          f'({texels/1e6:.1f} M texels, {100*texels/(len(pages)*a.page**2):.0f}% full)')

    # ---- render ----------------------------------------------------------
    expo = a.exposure

    # atlas UVs first: the Cycles scene needs them before anything is drawn
    _ = expo
    uvout = {}
    for ch in charts:
        grp = ch.prim
        P, idx = grp['P'], grp['T']
        for k in ch.tris:
            q = P[idx[k]]
            sc_ = (q - ch.org) @ ch.u
            tc_ = (q - ch.org) @ ch.v
            uu = (ch.x + (sc_ - ch.s0) / ch.texel) / a.page
            vv = (ch.y + (tc_ - ch.t0) / ch.texel) / a.page
            om, op, ok_ = grp['own'][int(k)]
            uvout.setdefault((om, op), {})[ok_] = (np.stack([uu, vv], 1), ch.page)

    cyc = None
    cyc_env = None
    if a.cycles:
        cyc, cyc_env = run_cycles(a, g, J, charts, uvout, bake_prims, pages,
                                  bsp, scale, shift)

    # A Source lightmap and a Cycles bake are not in the same units, and
    # exposing one as if it were the other is what turned the big outdoor
    # pages into blank white sheets. A lightmap value is absolute - 128 is
    # "fully lit" because the engine's overbright is 2 - whereas a Cycles
    # DIFFUSE pass with use_pass_color off is irradiance/pi in the scene's own
    # watts. On de_cpl_strike that is sky 129 + sun 829/pi, so "fully lit" is
    # 393, not 128: at 128 the median texel already tonemapped to 0.92 and 71%
    # of page 0 sat above it, which is exactly the blown-out look.
    ref = expo if expo is not None else LIGHT_FULL
    if expo is None and cyc is not None:
        ref = cycles_reference(cyc_env, cyc)
    knee = KNEE_AT
    lgamma = a.light_gamma
    if lgamma is None:
        lgamma = ((ZONE_ADAPT_GAMMA if adapt is not None else ZONE_NOSPEC_GAMMA)
                  if a.flat_normals else 1.0)
    fgain = a.flat_gain
    if fgain is None:
        fgain = parse_flat_normal(a.flat_normals)[1] if a.flat_normals else 1.0
    if False:
        if cyc is not None:
            v = np.concatenate([p[p.sum(2) > 0][:, :3].reshape(-1, 3)[::7]
                                for p in cyc.values() if (p.sum(2) > 0).any()]) \
                if any((p.sum(2) > 0).any() for p in cyc.values()) else None
            lm = (v @ np.array([.2126, .7152, .0722])) if v is not None else None
            ref = max(float(np.percentile(lm, EXPOSURE_PCT)), 1e-4) \
                if lm is not None and len(lm) else 1.0
        else:
            lum = []
            for rec in light.faces:
                vv = rec[4].reshape(-1, 3)
                lum.append(vv[::max(1, len(vv) // 32)]
                           @ np.array([.2126, .7152, .0722]))
            ref = max(
                float(np.percentile(np.concatenate(lum), EXPOSURE_PCT)), 1e-4)
    print(f'[3/5] light: fully lit at {ref:.0f}, brightness {a.brightness}, '
          f'flat gain {fgain:.2f}, gamma {lgamma} '
          f'({"Cycles watts" if cyc is not None else "Source lightmap units"})')
    if a.flat_normals:
        _fn = parse_flat_normal(a.flat_normals)[0]
        print(f'[3/5] flat normal {_fn[0]:+.3f},{_fn[1]:+.3f},{_fn[2]:+.3f}'
              f'  roughness {max(a.roughness_min, 0.04):.2f}'
              f'  blend {getattr(a, "normal_blend", 0.0)}'
              f'  (game multiplies the atlas by ~{1.0 / fgain:.2f})')

    imgs = ({p: np.zeros((a.page, a.page, 3), np.float32) for p in pages}
            if a.zone_v3 else
            {p: np.zeros((a.page, a.page, 4 if p in alpha_pages else 3),
                         np.uint8) for p in pages})
    mipcache = {}
    stats = collections.Counter()
    for ci, ch in enumerate(charts):
        grp = ch.prim
        P, UV, idx, matidx = grp['P'], grp['UV'], grp['T'], grp['mat']
        want_a = ch.page in alpha_pages
        key = (matidx if matidx is not None else -1, want_a)
        if a.zone_v3:
            pass                      # no albedo in a lightmap
        elif key not in mipcache:
            src = g.mat_albedo_image(matidx)
            if src is None:
                col = (J['materials'][matidx].get('pbrMetallicRoughness', {})
                       .get('baseColorFactor', [0.7, 0.7, 0.7, 1])
                       if matidx is not None else [0.7, 0.7, 0.7, 1])
                flat = np.zeros((2, 2, 4 if want_a else 3), np.float32)
                flat[:, :, :3] = np.array(col[:3], np.float32) * 255.0
                if want_a:
                    flat[:, :, 3] = 255.0
                mipcache[key] = [flat]
            else:
                data, _mime = g.image_bytes(src)
                import io
                mipcache[key] = mips_of(Image.open(io.BytesIO(data)), want_a)
        mips = mipcache.get(key)

        tw, th = ch.tw, ch.th
        acc_uv = np.zeros((th, tw, 2))
        acc_p = np.zeros((th, tw, 3))
        acc_n = np.zeros((th, tw, 3))
        mask = np.zeros((th, tw), bool)
        for k in ch.tris:
            tri = idx[k]
            q = P[tri]
            s = (q - ch.org) @ ch.u
            t = (q - ch.org) @ ch.v
            x = (s - ch.s0) / ch.texel
            y = (t - ch.t0) / ch.texel
            x0 = max(0, int(math.floor(x.min())) - 1)
            x1 = min(tw - 1, int(math.ceil(x.max())) + 1)
            y0 = max(0, int(math.floor(y.min())) - 1)
            y1 = min(th - 1, int(math.ceil(y.max())) + 1)
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5,
                                 np.arange(y0, y1 + 1) + 0.5)
            v0 = np.array([x[1] - x[0], y[1] - y[0]])
            v1 = np.array([x[2] - x[0], y[2] - y[0]])
            v2x = gx - x[0]; v2y = gy - y[0]
            den = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(den) >= 1e-12:
                bu = (v2x * v1[1] - v1[0] * v2y) / den
                bv = (v0[0] * v2y - v2x * v0[1]) / den
                e = 0.5 / max(tw, th)
                ins = (bu >= -e) & (bv >= -e) & (bu + bv <= 1 + e)
            if abs(den) < 1e-12 or not ins.any():
                # Smaller than a texel, or a sliver between texel centres: it
                # still gets the texel under its centre. Skipping these left
                # 30 % of gm_br_pitfalls' triangles - door slabs, skirting,
                # every high-poly prop - sampling the empty page: black.
                cx = min(tw - 1, max(0, int(x.mean())))
                cy = min(th - 1, max(0, int(y.mean())))
                if not mask[cy, cx]:
                    fn = np.cross(q[1] - q[0], q[2] - q[0])
                    fl = np.linalg.norm(fn)
                    acc_uv[cy, cx] = UV[tri].mean(0)
                    acc_p[cy, cx] = q.mean(0)
                    acc_n[cy, cx] = fn / fl if fl > 1e-12 else ch.n
                    mask[cy, cx] = True
                continue
            w0 = (1 - bu - bv)[ins]; w1 = bu[ins]; w2 = bv[ins]
            tuv = (UV[tri[0]][None, :] * w0[:, None] + UV[tri[1]][None, :] * w1[:, None]
                   + UV[tri[2]][None, :] * w2[:, None])
            ys = (gy[ins] - 0.5).astype(int); xs = (gx[ins] - 0.5).astype(int)
            tp = (q[0][None, :] * w0[:, None] + q[1][None, :] * w1[:, None]
                  + q[2][None, :] * w2[:, None])
            fn = np.cross(q[1] - q[0], q[2] - q[0])
            fl = np.linalg.norm(fn)
            acc_uv[ys, xs] = tuv
            acc_p[ys, xs] = tp
            acc_n[ys, xs] = fn / fl if fl > 1e-12 else ch.n
            mask[ys, xs] = True
        if not mask.any():
            continue
        wsel = acc_p[mask]
        sel = mask
        if a.zone_v3:
            if cyc is not None:
                li = cyc[ch.page][ch.y:ch.y + th, ch.x:ch.x + tw][mask]
                stats['exact'] += int(mask.sum())
            else:
                nsel = acc_n[mask]
                li = np.zeros((len(wsel), 3))
                qn = np.round(nsel * 100).astype(np.int64)
                uq, inv = np.unique(qn, axis=0, return_inverse=True)
                for gi in range(len(uq)):
                    m = inv == gi
                    li[m] = light.sample(to_src(wsel[m]), dir_to_src(nsel[m][0]), stats)
            # the game's units: 1.0 shows the albedo as it is - what Source's
            # 128 ("fully lit") means, and Cycles' fully-lit reference
            out = np.zeros((th, tw, 3), np.float32)
            # the same 3 % pedestal as the albedo path: a face whose light lookup
            # finds nothing comes out dim, not a black hole
            out[sel] = np.maximum(li[:, :3], a.shadow_pedestal * ref) * (a.brightness / ref)
            out = dilate(out, mask, rounds=8)
            page = imgs[ch.page]
            page[ch.y:ch.y + th, ch.x:ch.x + tw] = out
            for dy in range(1, PAD + 1):
                if ch.y - dy >= 0:
                    page[ch.y - dy, ch.x:ch.x + tw] = out[0]
                if ch.y + th + dy - 1 < a.page:
                    page[ch.y + th + dy - 1, ch.x:ch.x + tw] = out[-1]
                if ch.x - dy >= 0:
                    page[ch.y - PAD:ch.y + th + PAD, ch.x - dy] = page[ch.y - PAD:ch.y + th + PAD, ch.x]
                if ch.x + tw + dy - 1 < a.page:
                    page[ch.y - PAD:ch.y + th + PAD, ch.x + tw + dy - 1] = \
                        page[ch.y - PAD:ch.y + th + PAD, ch.x + tw - 1]
            if (ci + 1) % 500 == 0:
                print(f'      {ci+1}/{len(charts)} charts')
            continue
        # albedo, at the mip matching this chart's density
        tex0 = mips[0]
        # How many source texels one atlas texel covers. Take it from the
        # chart's own (s,t)->(u,v) Jacobian: differencing the tile instead
        # walks over the uncovered texels, reads a huge gradient at every
        # chart edge and pins the whole thing to the 1x1 mip - which is what
        # turned the first render into coloured mush.
        kbig = ch.tris[int(np.argmax(np.linalg.norm(np.cross(
            P[idx[ch.tris]][:, 1] - P[idx[ch.tris]][:, 0],
            P[idx[ch.tris]][:, 2] - P[idx[ch.tris]][:, 0]), axis=1)))]
        q3 = P[idx[kbig]]
        s3 = (q3 - ch.org) @ ch.u
        t3 = (q3 - ch.org) @ ch.v
        A = np.array([[s3[1] - s3[0], t3[1] - t3[0]],
                      [s3[2] - s3[0], t3[2] - t3[0]]])
        duv = np.array([UV[idx[kbig][1]] - UV[idx[kbig][0]],
                        UV[idx[kbig][2]] - UV[idx[kbig][0]]])
        try:
            Jm = np.linalg.solve(A, duv)          # rows: d(uv)/ds, d(uv)/dt
        except np.linalg.LinAlgError:
            Jm = np.zeros((2, 2))
        px = np.abs(Jm) * np.array([tex0.shape[1], tex0.shape[0]])[None, :]
        foot = max(float(px.max()) * ch.texel, 1.0)
        lvl = int(np.clip(round(math.log2(foot)), 0, len(mips) - 1))
        alb = sample_tex(mips[lvl], acc_uv[sel][:, 0], acc_uv[sel][:, 1])
        if cyc is not None:
            # Cycles already baked into this exact tile - no lookup at all
            li = cyc[ch.page][ch.y:ch.y + th, ch.x:ch.x + tw][mask]
            stats['exact'] += int(mask.sum())
        else:
            # the light query is keyed on the TRIANGLE's normal, not the
            # chart's average: a chart is allowed 20 degrees of curvature, and
            # a face only matches its BSP original within about 8
            nsel = acc_n[mask]
            li = np.zeros((len(wsel), 3))
            qn = np.round(nsel * 100).astype(np.int64)
            uq, inv = np.unique(qn, axis=0, return_inverse=True)
            for gi in range(len(uq)):
                m = inv == gi
                li[m] = light.sample(to_src(wsel[m]),
                                     dir_to_src(nsel[m][0]), stats)
        nch = 4 if ch.page in alpha_pages else 3
        out = np.zeros((th, tw, nch), np.float32)
        if nch == 4:
            out[:, :, 3] = 255.0
        a_chan = alb[:, 3] if alb.shape[1] == 4 else None
        alb = alb[:, :3]
        if adapt is not None and cyc is None:
            li = li * adapt.gain(to_src(wsel))[:, None]
        val = _srgb_to_lin(alb) * shadow_floor(
            tonemap(li, ref, a.brightness, knee), a.shadow_pedestal)
        if fgain != 1.0:
            # The gain belongs on the finished pixel, not on the light. The
            # game multiplies ALBEDO by a flat ~0.45, so what has to come back
            # up is albedo x light, not light alone - putting it on the light
            # only pushes the tone curve into its shoulder and washes the
            # lighting out while leaving the map just as dark.
            val = tonemap(val, 1.0, fgain, KNEE_AT)
        if lgamma != 1.0:
            val = np.power(np.clip(val, 0.0, 1.0), lgamma)
        rgb = _lin_to_srgb(np.clip(val, 0, 1)).astype(np.float32)
        if nch == 4:
            # the cut-out is the source texture's own alpha, straight through:
            # lighting a leaf must not change its shape
            ac = a_chan if a_chan is not None else np.full(len(rgb), 255.0)
            out[sel] = np.concatenate([rgb, ac[:, None]], axis=1)
        else:
            out[sel] = rgb
        out = dilate(out, mask)
        page = imgs[ch.page]
        page[ch.y:ch.y + th, ch.x:ch.x + tw] = out.astype(np.uint8)
        for dy in range(1, PAD + 1):
            if ch.y - dy >= 0:
                page[ch.y - dy, ch.x:ch.x + tw] = out[0]
            if ch.y + th + dy - 1 < a.page:
                page[ch.y + th + dy - 1, ch.x:ch.x + tw] = out[-1]
            if ch.x - dy >= 0:
                page[ch.y - PAD:ch.y + th + PAD, ch.x - dy] = \
                    page[ch.y - PAD:ch.y + th + PAD, ch.x]
            if ch.x + tw + dy - 1 < a.page:
                page[ch.y - PAD:ch.y + th + PAD, ch.x + tw + dy - 1] = \
                    page[ch.y - PAD:ch.y + th + PAD, ch.x + tw - 1]
        # atlas UVs for this chart's triangles
        if (ci + 1) % 500 == 0:
            print(f'      {ci+1}/{len(charts)} charts')
    tot = stats['exact'] + stats['inexact']
    if cyc is not None:
        print(f'[4/5] light: Cycles, {len(cyc)} pages, '
              f'{stats["exact"]/1e6:.1f} M texels lit')
    else:
        print(f'[4/5] light lookups: {100*stats["exact"]/max(tot,1):.1f}% exact, '
              f'{stats["near"]} nearest-surface, {stats["grid"]} grid')
    if cyc is None and tot and stats['exact'] / tot < 0.25:
        print('      ! most of the map found no BSP face under it - the origin '
              'shift is probably wrong. Pass --shift or the right --report.')
    if a.zone_v3:
        write_v3(a, g, imgs, pages, uvout, bake_prims)
    else:
        write_glb(a, g, imgs, pages, uvout, bake_prims)
    print(f'[5/5] done in {time.time()-t0:.1f}s')


def find_blender(explicit=None):
    import shutil
    if explicit:
        return explicit
    hit = shutil.which('blender')
    if hit:
        return hit
    pats = [r'C:\Program Files\Blender Foundation\Blender *\blender.exe',
            r'C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe',
            '/Applications/Blender.app/Contents/MacOS/Blender',
            '/usr/bin/blender', '/usr/local/bin/blender']
    import glob as _g
    got = []
    for pat in pats:
        got += _g.glob(pat)
    if not got:
        raise SystemExit(
            'could not find Blender. Pass --blender "<path to blender.exe>".')
    got.sort()
    return got[-1]          # the newest version found


def copy_materials(g, out, rough_min=0.0):
    """Copy every image and material from the input .glb into `out`.

    `rough_min` floors roughness on the way through and zeroes metallic: both
    feed the specular lobe the game's own baked maps never get.
    """
    J = g.j
    img_map, mat_map = {}, {}

    def copy_image(i):
        if i in img_map:
            return img_map[i]
        data, mime = g.image_bytes(i)
        v = out._view(data)
        out.j['images'].append({'bufferView': v, 'mimeType': mime,
                                'name': J['images'][i].get('name', f'img{i}')})
        out.j['textures'].append({'sampler': 0,
                                  'source': len(out.j['images']) - 1})
        img_map[i] = len(out.j['textures']) - 1
        return img_map[i]

    for mi, m in enumerate(J.get('materials', [])):
        pbr = m.get('pbrMetallicRoughness', {})
        alb = pbr.get('baseColorTexture')
        nrm = m.get('normalTexture')
        ai = copy_image(J['textures'][alb['index']]['source']) if alb else None
        ni = copy_image(J['textures'][nrm['index']]['source']) if nrm else None
        mat_map[mi] = out.material(
            m.get('name', f'mat{mi}'), albedo_tex=ai, normal_tex=ni,
            roughness=max(float(pbr.get('roughnessFactor', 0.85)), rough_min),
            metallic=(0.0 if rough_min > 0
                      else float(pbr.get('metallicFactor', 0.0))),
            base_color=tuple(pbr.get('baseColorFactor', (1, 1, 1, 1))),
            alpha_mode=m.get('alphaMode', 'OPAQUE'),
            double_sided=bool(m.get('doubleSided', False)),
            emissive=tuple(m.get('emissiveFactor')) if m.get('emissiveFactor')
            else None)
        if 'KHR_materials_unlit' in (m.get('extensions') or {}):
            out.j['materials'][mat_map[mi]]['extensions'] = {'KHR_materials_unlit': {}}
            used = out.j.setdefault('extensionsUsed', [])
            if 'KHR_materials_unlit' not in used:
                used.append('KHR_materials_unlit')
    return mat_map


def run_cycles(a, g, J, charts, uvout, bake_prims, pages, bsp, scale, shift):
    """Write a scene, bake it in Blender, and read the light pages back."""
    import subprocess, tempfile
    from zonemap import srclights

    if a.reuse_bake:
        cyc = {}
        for p in pages:
            f = os.path.join(a.reuse_bake, f'page{p}.npy')
            if not os.path.isfile(f):
                raise SystemExit(
                    f'--reuse-bake {a.reuse_bake}: no page{p}.npy. The chart '
                    'layout must match the run that produced it (same '
                    '--texel/--budget/--page).')
            # mmap, not load: 21 pages of 2048^2 float32 is a gigabyte, and
            # numpy short-reads rather than raising when it cannot get it -
            # which surfaces as "cannot reshape array of size 2057184".
            # The rasterise only ever reads one chart's slice at a time.
            cyc[p] = np.load(f, mmap_mode='r')
        print(f'[3/5] reusing {len(cyc)} light pages from {a.reuse_bake}')
        _, env = srclights.collect(bsp, scale=scale, shift=shift,
                                   point_scale=a.point_scale,
                                   sun_scale=a.sun_scale) if bsp else (None, None)
        return cyc, env

    work = (os.path.splitext(a.out or a.glb)[0] + '_cycles') if a.keep_scene \
        else tempfile.mkdtemp(prefix='zonebake_')
    os.makedirs(work, exist_ok=True)
    scene_path = os.path.join(work, 'scene.glb')

    out = Glb(generator='zone-map-forge cycles scene')
    mat_map = copy_materials(g, out, a.roughness_min)
    # The bake runs with use_pass_color off, so a page object's albedo is never
    # read - but its ALPHA is: an alpha-blended bake target comes back empty,
    # which would black out every tree. One opaque material for all of them.
    bake_mat = out.material('lm_bake_target', roughness=1.0, metallic=0.0)

    # one object per page: the bake target has to own a UV set that is inside
    # 0-1 for exactly one image, and splitting by page is the cheapest way
    by_page = collections.defaultdict(lambda: collections.defaultdict(list))
    for ch in charts:
        grp = ch.prim
        for k in ch.tris:
            om, op, ok_ = grp['own'][int(k)]
            uv3, page = uvout[(om, op)][ok_]
            by_page[page][grp['mat']].append((grp['P'][grp['T'][k]],
                                              grp['UV'][grp['T'][k]], uv3))
    nodes = []
    for page, mats in sorted(by_page.items()):
        prims = []
        for mat, tris in mats.items():
            pos = np.concatenate([t[0] for t in tris]).astype(np.float32)
            uv0 = np.concatenate([t[1] for t in tris]).astype(np.float32)
            uv1 = np.concatenate([t[2] for t in tris]).astype(np.float32)
            nrm = np.zeros_like(pos)
            tp = pos.reshape(-1, 3, 3)
            fn = np.cross(tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0])
            fl = np.linalg.norm(fn, axis=1, keepdims=True)
            fn = fn / np.where(fl == 0, 1, fl)
            nrm = np.repeat(fn, 3, axis=0).astype(np.float32)
            idx = np.arange(len(pos), dtype=np.uint32).reshape(-1, 3)
            prims.append((pos, nrm, uv0, idx, bake_mat, uv1))
        nodes.append(out.node(f'bakepage_{page}', mesh=out.mesh(f'bakepage_{page}',
                                                                prims)))
    # occluders: everything the bake does not target still has to cast shadows
    nocc = 0
    if not a.no_occluders:
        users = collections.Counter(n['mesh'] for n in J['nodes'] if 'mesh' in n)
        wm = {}

        def walk(i, M):
            Mi = M @ node_matrix(J['nodes'][i])
            wm[i] = Mi
            for c in J['nodes'][i].get('children', ()):
                walk(c, Mi)
        for rt in J['scenes'][J.get('scene', 0)]['nodes']:
            walk(rt, np.eye(4))
        baked_meshes = {b[0] for b in bake_prims}
        for ni, n in enumerate(J['nodes']):
            mi = n.get('mesh')
            if mi is None or mi in baked_meshes or SKY.is_sky_node(n) or EX.is_sky3d_node(n):
                continue        # the sky surfaces must not shade the map from the sky
            M = wm.get(ni, np.eye(4))
            prims = []
            for pr in J['meshes'][mi]['primitives']:
                at = pr['attributes']
                if 'POSITION' not in at:
                    continue
                P = g.acc(at['POSITION']).astype(np.float64)
                P = (P @ M[:3, :3].T + M[:3, 3]).astype(np.float32)
                nr = (g.acc(at['NORMAL']).astype(np.float64) @ M[:3, :3].T
                      ).astype(np.float32) if 'NORMAL' in at else \
                    np.tile(np.array([0, 1, 0], np.float32), (len(P), 1))
                uv = (g.acc(at['TEXCOORD_0']).astype(np.float32)
                      if 'TEXCOORD_0' in at else np.zeros((len(P), 2), np.float32))
                idx = (g.acc(pr['indices']).astype(np.uint32).reshape(-1, 3)
                       if 'indices' in pr else
                       np.arange(len(P), dtype=np.uint32).reshape(-1, 3))
                prims.append((P, nr, uv, idx,
                              mat_map.get(pr.get('material'), 0)))
            if prims:
                nodes.append(out.node(f'occ_{ni}',
                                      mesh=out.mesh(f'occ_{ni}', prims)))
                nocc += 1
    root = out.node('scene', children=nodes)
    out.root(root)
    mb = out.save(scene_path) / 1e6
    print(f'[3/5] cycles scene: {len(by_page)} page objects, {nocc} occluders, '
          f'{mb:.0f} MB -> {scene_path}')

    lights, env = srclights.collect(bsp, scale=scale, shift=shift,
                                    point_scale=a.point_scale,
                                    sun_scale=a.sun_scale)
    R = getattr(a, '_turn_R', np.eye(3))     # the map's turn (map_turn_deg)
    for rec in lights + ([env] if env else []):
        for k in ('pos', 'dir'):
            if k in rec:
                rec[k] = (R @ np.asarray(rec[k], np.float64)).tolist()
    lj = os.path.join(work, 'lights.json')
    json.dump({'lights': lights, 'env': env, 'page_px': a.page,
               'bounces': a.bounces}, open(lj, 'w'))
    print(f'[3/5] lights: {len(lights)} point/spot'
          + (', sun' if env else ', no light_environment'))

    exe = find_blender(a.blender)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'bake_cycles_blender.py')
    # --factory-startup: load none of the user's add-ons or preferences. A
    # real Blender install has both, and in background mode they misbehave -
    # this one raised "GPU functions for drawing requires the gpu module to be
    # initialized" at startup and then a handler on depsgraph_update_post that
    # threw AttributeError: 'Scene' object has no attribute 'vs' on EVERY
    # depsgraph update, which during a bake is constant. Nothing here needs a
    # user add-on: glTF import and Cycles are both built in, and the GPU
    # device is selected explicitly rather than read from preferences.
    cmd = [exe, '-b', '--factory-startup', '--disable-autoexec',
           '-P', script, '--', scene_path, lj, work,
           '--samples', str(a.samples)]
    print(f'[3/5] {os.path.basename(exe)} baking {len(by_page)} pages at '
          f'{a.samples} samples, {a.bounces} bounces - this is the slow part.\n'
          f'      Each page prints when it finishes; if the gap between pages '
          f'is minutes, cut --samples or --bounces.', flush=True)
    # Stream it. Capturing meant no sign of life for however long the bake
    # took - and once the lights were fixed and rays actually started bouncing,
    # that became long enough to look like a hang. Read bytes, not text:
    # Blender writes bytes Windows' cp1252 cannot decode, and with text=True
    # that failure lands in subprocess's reader THREAD, so the bake goes on
    # while stdout comes back empty and the traceback looks like a crash.
    t_b = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=0)
    tail = collections.deque(maxlen=40)
    for raw_line in iter(proc.stdout.readline, b''):
        line = raw_line.decode('utf-8', 'replace').rstrip()
        tail.append(line)
        if line.startswith('[cycles]') or 'Error' in line or 'error' in line[:40]:
            print(f'      [{time.time()-t_b:5.0f}s] {line}', flush=True)
    proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        print('\n'.join(tail))
        raise SystemExit(f'blender exited {rc}')
    cyc = {}
    for p in pages:
        f = os.path.join(work, f'page{p}.npy')
        if not os.path.isfile(f):
            raise SystemExit(f'blender produced no page {p} ({f})')
        cyc[p] = np.load(f, mmap_mode='r' if a.keep_scene else None)
    if not a.keep_scene:
        import shutil as _sh
        _sh.rmtree(work, ignore_errors=True)
    return cyc, env


def cycles_reference(env, cyc):
    """"Fully lit", in the units a Cycles DIFFUSE pass comes back in.

    The pass is irradiance / pi - the number the albedo gets multiplied by -
    so a surface under an open sky of radiance S reads S, and one facing the
    sun head-on reads sun_irradiance / pi on top of it. That sum is the
    brightest a diffuse surface gets from the environment, which is what the
    tone curve should call 1.0. Measured against the real bake: page 0 of
    de_cpl_strike peaks at 366.6 and 129 + 829/pi = 393.
    """
    if env and env.get('type') == 'sun':
        # both terms / PI: the sun because the pass is irradiance/PI, and the
        # sky because the world is now fed as radiance (amb / PI), so a surface
        # open to the whole sky reads amb / PI in the pass, not amb.
        return max((float(env.get('energy', 0.0))
                    + float(env.get('ambient', 0.0))) / math.pi, 1e-4)
    # no light_environment (an indoor map): nothing absolute to anchor on, so
    # fall back on the map's own bright end
    lit = [p[p.sum(2) > 0][::17] for p in cyc.values() if (p.sum(2) > 0).any()]
    if not lit:
        return LIGHT_FULL
    v = np.concatenate(lit) @ np.array([.2126, .7152, .0722])
    return max(float(np.percentile(v, EXPOSURE_PCT)), 1e-4)


PRESET_NORMALS = {'auto': (ZONE_FLAT_NORMAL, ZONE_FLAT_GAIN),
                  'default': (ZONE_FLAT_NORMAL, ZONE_FLAT_GAIN),
                  '': (ZONE_FLAT_NORMAL, ZONE_FLAT_GAIN),
                  'nospec': (ZONE_NOSPEC_NORMAL, ZONE_NOSPEC_GAIN),
                  'low': (ZONE_LOW_NORMAL, ZONE_LOW_GAIN),
                  'mid': (ZONE_MID_NORMAL, ZONE_MID_GAIN)}


def parse_flat_normal(spec):
    """'auto' | 'nospec' | 'x,y,z' -> (unit normal, its default gain)."""
    key = str(spec).strip().lower()
    if key in PRESET_NORMALS:
        v, gain = PRESET_NORMALS[key]
        return np.array(v, np.float32), gain
    v = np.array([float(x) for x in str(spec).split(',')], np.float32)
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return np.array(ZONE_FLAT_NORMAL, np.float32), ZONE_FLAT_GAIN
    return v / n, ZONE_FLAT_GAIN


def fill_empty(img, mask):
    """Every texel outside `mask` takes the light of the nearest baked ones
    (pull-push: average down a pyramid, then fill back up), so mips and the
    packer's halving never average black in."""
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


def write_exr(path, rgb):
    """an uncompressed float RGB OpenEXR (top row first), for the Godot packer"""
    h, w = rgb.shape[:2]

    def attr(name, typ, data):
        return name.encode() + b'\0' + typ.encode() + b'\0' + struct.pack('<i', len(data)) + data
    ch = b''.join(c.encode() + b'\0' + struct.pack('<iBBBBii', 2, 0, 0, 0, 0, 1, 1) for c in 'BGR') + b'\0'
    box = struct.pack('<iiii', 0, 0, w - 1, h - 1)
    hdr = (struct.pack('<ii', 20000630, 2) + attr('channels', 'chlist', ch) +
           attr('compression', 'compression', b'\0') + attr('dataWindow', 'box2i', box) +
           attr('displayWindow', 'box2i', box) + attr('lineOrder', 'lineOrder', b'\0') +
           attr('pixelAspectRatio', 'float', struct.pack('<f', 1.0)) +
           attr('screenWindowCenter', 'v2f', struct.pack('<ff', 0, 0)) +
           attr('screenWindowWidth', 'float', struct.pack('<f', 1.0)) + b'\0')
    row = w * 3 * 4
    start = len(hdr) + 8 * h
    offs = np.arange(h, dtype=np.uint64) * (8 + row) + start
    planar = np.ascontiguousarray(rgb[:, :, ::-1].transpose(0, 2, 1).astype('<f4'))  # rows of B, G, R
    lines = np.empty((h, 8 + row), np.uint8)
    lines[:, :4] = np.arange(h, dtype='<i4').view(np.uint8).reshape(h, 4)
    lines[:, 4:8] = np.full(h, row, '<i4').view(np.uint8).reshape(h, 4)
    lines[:, 8:] = planar.reshape(h, -1).view(np.uint8)
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(offs.astype('<u8').tobytes())
        f.write(lines.tobytes())


def find_godot(explicit=None):
    if explicit:
        return explicit
    import glob
    home = os.path.expanduser('~')
    found = []
    for root in (os.path.join(home, 'Downloads'), os.path.join(home, 'Desktop'), r'C:\Program Files\Godot'):
        for depth in ('', '*', '*/*'):
            found += glob.glob(os.path.join(root, depth, 'Godot_v4.*_win64*.exe'))
            found += glob.glob(os.path.join(root, depth, 'Godot_v4.*_linux.x86_64'))
    found = [f for f in found if os.path.isfile(f) and not re.search(r'v4\.[0-3][.-]', f)]
    found.sort(key=lambda f: ('console' not in f, f))
    return found[0] if found else None


def write_v3(a, g, light_pages, pages, uvout, bake_prims):
    """The beta's real-lightmap map folder: one node per light page (its triangles
    with their own materials, UV2 into the page), the light as EXR, the textures
    as they are; then tools/pack_zone_map.gd packs it."""
    import shutil, subprocess
    from zonemap.convert import slug
    J = g.j
    out = Glb(generator='zone-map-forge bake_glb --zone-v3')
    mat_map = copy_materials(g, out, a.roughness_min)
    prim_src = {(b[0], b[1]): b for b in bake_prims}
    # every baked triangle onto its light page, with its own material
    by_page = collections.defaultdict(lambda: collections.defaultdict(list))
    for (mi, pi), tris_ in uvout.items():
        _mi, _pi, P, N, UV, idx, matidx = prim_src[(mi, pi)]
        ns = (mi, pi) in NS_PRIMS
        for k, (uv3, page) in tris_.items():
            by_page[page][(matidx, ns)].append((idx[k], uv3, P, N, UV))
    page_nodes = []
    collider_nodes = []
    page_uv1 = collections.defaultdict(list)
    for p in sorted(by_page):
        prims = []
        prims_ns = []
        for (matidx, ns), items in by_page[p].items():
            tri = [t for t, _, _, _, _ in items]
            pos = np.concatenate([P[t] for t, _, P, _, _ in items]).astype(np.float32)
            uv0 = np.concatenate([UV[t] for t, _, _, _, UV in items]).astype(np.float32)
            uv1 = np.concatenate([u for _, u, _, _, _ in items]).astype(np.float32)
            if all(N is not None for _, _, _, N, _ in items):
                nrm = np.concatenate([N[t] for t, _, _, N, _ in items]).astype(np.float32)
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
            else:
                q = pos.reshape(-1, 3, 3)
                fn = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
                fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
                nrm = np.repeat(fn, 3, axis=0).astype(np.float32)
            page_uv1[p].append(uv1)
            ii = np.arange(len(pos), dtype=np.uint32).reshape(-1, 3)
            (prims_ns if ns else prims).append((pos, nrm, uv0, ii, mat_map.get(matidx), uv1))
        # Godot gives a mesh at most MAX_SURFACES surfaces (one per material) and
        # drops the rest as it loads the .glb, with only a log line for each: a
        # busy page has to be several nodes, sharing its lightmap. gm_br_pitfalls
        # lost 1,435 materials' triangles that way, door slabs among them
        if prims:
            parts = -(-len(prims) // MAX_SURFACES)
            size = -(-len(prims) // parts)
            for k in range(parts):
                name = f'lm{p:03d}_zlms-{a.page}' if k == 0 else f'lm{p:03d}s{k}_zlms-{a.page}'
                page_nodes.append((p, out.node(name, mesh=out.mesh(name, prims[k * size:(k + 1) * size])),
                                   name))
        if prims_ns:
            # never-solid brushes (zonemap.build.never_solid) on this page: a
            # node of their own sharing the page's light, and a '-collider'
            # stand-in so the game makes nothing of them solid
            parts = -(-len(prims_ns) // MAX_SURFACES)
            size = -(-len(prims_ns) // parts)
            for k in range(parts):
                name = f'lm{p:03d}n{k or ""}_zlms-{a.page}'
                part = prims_ns[k * size:(k + 1) * size]
                page_nodes.append((p, out.node(name, mesh=out.mesh(name, part)), name))
                p0 = part[0][0][0]
                cp = np.array([p0, p0 + np.float32([0.02, 0, 0]), p0 + np.float32([0, 0.02, 0])], np.float32)
                cm = out.mesh(name + '-collider', [(cp, np.tile(np.float32([0, 0, 1]), (3, 1)),
                                                    np.zeros((3, 2), np.float32),
                                                    np.array([[0, 1, 2]], np.uint32), part[0][4], None)])
                collider_nodes.append(out.node(name + '-collider', mesh=cm))
    # everything not baked (instanced meshes, skipped prims) as it was - bar
    # the forge's collider stand-ins, made again above per page node
    stub_nodes = {ni for ni, n in enumerate(J['nodes']) if (n.get('name') or '').endswith('-collider')}
    stub_meshes = {J['nodes'][ni].get('mesh') for ni in stub_nodes}
    mesh_map = {}
    for mi, m in enumerate(J['meshes']):
        if mi in stub_meshes:
            continue
        prims = []
        for pi, pr in enumerate(m['primitives']):
            if (mi, pi) in uvout:
                continue
            at = pr['attributes']
            pos = g.acc(at['POSITION']).astype(np.float32)
            nrm = g.acc(at['NORMAL']).astype(np.float32) if 'NORMAL' in at else None
            uv = (g.acc(at['TEXCOORD_0']).astype(np.float32) if 'TEXCOORD_0' in at
                  else np.zeros((len(pos), 2), np.float32))
            idx = (g.acc(pr['indices']).astype(np.uint32).reshape(-1, 3) if 'indices' in pr
                   else np.arange(len(pos), dtype=np.uint32).reshape(-1, 3))
            col = g.acc(at['COLOR_0']).astype(np.float32) if 'COLOR_0' in at else None
            prims.append((pos, nrm, uv, idx, mat_map.get(pr.get('material')), None, col))
        if prims:
            mesh_map[mi] = out.mesh(m.get('name', f'mesh{mi}'), prims)
    node_map = {}
    for ni, n in enumerate(J['nodes']):
        if ni in stub_nodes:
            continue
        t = r = s = None
        if any(k in n for k in ('matrix', 'translation', 'rotation', 'scale')):
            t, r, s = decompose(node_matrix(n))
        node_map[ni] = out.node(n.get('name', f'n{ni}'), mesh=mesh_map.get(n.get('mesh')),
                                translation=t, rotation=r, scale=s)
        if n.get('extras'):
            out.j['nodes'][node_map[ni]]['extras'] = n['extras']
    for ni, n in enumerate(J['nodes']):
        kids = [node_map[c] for c in n.get('children', ()) if c in node_map]
        if kids:
            out.j['nodes'][node_map[ni]]['children'] = kids
    for rt in J['scenes'][J.get('scene', 0)]['nodes']:
        out.root(node_map[rt])
    for _, nd, _ in page_nodes:
        out.root(nd)
    for nd in collider_nodes:
        out.root(nd)
    if collider_nodes:
        print(f'[5/5] {len(collider_nodes)} never-solid page nodes, each with a -collider stand-in')
    # names the game cannot swap for its own library materials (it matches by
    # substring), unique, as forge.py names them
    seen = {}
    for mm in out.j['materials']:
        if not mm['name'].startswith(('z_', 'src_', 'zone_')):
            mm['name'] = 'src_' + slug(mm['name'])
        if mm['name'] in seen:
            seen[mm['name']] += 1
            mm['name'] = f"{mm['name']}_{seen[mm['name']]}"
        else:
            seen[mm['name']] = 0

    base = os.path.splitext(os.path.basename(a.out or a.glb))[0]
    name = base if base.endswith('_v3') else base + '_v3'
    work = os.path.join(os.path.dirname(os.path.abspath(a.out or a.glb)), name + '_zone_v3')
    for d in ('lightmaps', 'textures'):
        os.makedirs(os.path.join(work, d), exist_ok=True)
    out.save(os.path.join(work, 'map.glb'))
    # how many triangles sit on a texel nothing was baked into (black in the
    # game), then light for the gaps between charts: the packer halves each
    # page for the lower quality tiers and the GPU mips it, and an empty
    # gutter averaged in darkens every chart's rim
    dark = tris_n = 0
    for p in sorted({p for p, _, _ in page_nodes}):
        img = light_pages[p]
        filled = img.max(2) > 0
        for uv1 in page_uv1[p]:
            c = uv1.reshape(-1, 3, 2).mean(1)
            xs = np.clip((c[:, 0] * a.page).astype(np.int64), 0, a.page - 1)
            ys = np.clip((c[:, 1] * a.page).astype(np.int64), 0, a.page - 1)
            dark += int((~filled[ys, xs]).sum())
            tris_n += len(c)
        light_pages[p] = fill_empty(img, filled)
    print(f'[5/5] {100.0 * dark / max(tris_n, 1):.1f} % of {tris_n} triangles on unbaked texels '
          f'(given their neighbours\' light)')
    for p in sorted({p for p, _, _ in page_nodes}):
        write_exr(os.path.join(work, 'lightmaps', f'lm{p:03d}_zlms-{a.page}.exr'), light_pages[p])
    # what the game culls against (zonemap/occluder.py): the big solid
    # surfaces of the world, and the sky surfaces
    ov, oi, okk = OCC.from_glb(g.j, g.acc)
    OCC.write(os.path.join(work, 'occluder.bin'), ov, oi)
    ext = []
    for p in sorted({p for p, _, _ in page_nodes}):
        allp = np.concatenate([P[t].reshape(-1, 3) for items in by_page[p].values()
                               for t, _, P, _, _ in items])
        ext.append(allp.max(0) - allp.min(0))
    print(f'[5/5] occluders: {len(oi) // 3:,} triangles ({okk.get("world", 0):,} of walls and floors, '
          f'{okk.get("sky", 0):,} of sky); light pages span a median '
          f'{" x ".join(str(int(v)) for v in np.median(np.array(ext), 0))} m each')
    # textures, as the materials use them
    mats = {}
    tex_files = {}
    # the sky surfaces' material stays out of the bundle, so the game leaves
    # it the .glb's own (unshaded) for the map_extras mod to replace - and so
    # do the 3D skybox room's (unshaded, their textures in the .glb)
    sky_mats = {pr.get('material') for n in out.j['nodes']
                if (SKY.is_sky_node(n) or EX.is_sky3d_node(n)) and 'mesh' in n
                for pr in out.j['meshes'][n['mesh']]['primitives']}
    for k_mm, mm in enumerate(out.j['materials']):
        if k_mm in sky_mats:
            continue
        pbr = mm.get('pbrMetallicRoughness', {})
        e = {'color': [float(x) for x in pbr.get('baseColorFactor', [1, 1, 1, 1])],
             'roughness': float(pbr.get('roughnessFactor', 0.85)),
             'metallic': float(pbr.get('metallicFactor', 0.0)),
             'alpha': mm.get('alphaMode', 'OPAQUE') != 'OPAQUE',
             'albedo': None, 'emission_color': None, 'emission_energy': 0.0}
        t = pbr.get('baseColorTexture')
        if t is not None:
            src = out.j['textures'][t['index']]['source']
            if src not in tex_files:
                im = out.j['images'][src]
                v = out.j['bufferViews'][im['bufferView']]
                data = bytes(out.blob[v['byteOffset']:v['byteOffset'] + v['byteLength']])
                ext = '.jpg' if im.get('mimeType') == 'image/jpeg' else '.png'
                fn = f'textures/t{src}{ext}'
                with open(os.path.join(work, fn), 'wb') as fh:
                    fh.write(data)
                tex_files[src] = fn
            e['albedo'] = tex_files[src]
        if mm.get('emissiveFactor') and max(mm['emissiveFactor']) > 0:
            e['emission_color'] = [float(x) for x in mm['emissiveFactor']] + [1.0]
            e['emission_energy'] = 1.0
        mats[mm['name']] = e
    man = {}
    mp = os.path.join(os.path.dirname(os.path.abspath(a.glb)), 'manifest.json')
    if os.path.isfile(mp):
        man = json.load(open(mp))
    lv = man.get('lighting_variations') or {}
    var, cfg = (next(iter(lv.items())) if lv else
                ('day', {'skybox': man.get('skybox') or {'skybox_name': 'cloudy_sunny',
                                                         'sky_rotation': 0.0, 'energy': 1.0}}))
    cfg = {'skybox': {'skybox_name': cfg['skybox'].get('skybox_name', 'cloudy_sunny'),
                      'sky_rotation': float(cfg['skybox'].get('sky_rotation', 0.0)),
                      'energy': float(cfg['skybox'].get('energy', 1.0))}}
    info = {'version': 1, 'map': name, 'variation': var, 'lighting_variation': cfg,
            'spawn_point': man.get('spawn_point', [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            'objects': [dict(name=nm, size=a.page,
                             **({} if nm == f'lm{p:03d}_zlms-{a.page}'
                                else {'lightmap': f'lm{p:03d}_zlms-{a.page}'}))
                        for p, _, nm in page_nodes],
            'materials': mats, 'light': 'Cycles' if a.cycles else "Valve's lightmaps"}
    with open(os.path.join(work, 'bake_info.json'), 'w') as fh:
        json.dump(info, fh, indent=1)
    gd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pack_zone_map.gd')
    shutil.copy2(gd, os.path.join(work, 'pack_zone_map.gd'))
    npg = len({p for p, _, _ in page_nodes})
    nsplit = sum(1 for _, _, nm in page_nodes if not re.match(r'^lm\d+n\d*_', nm))
    print(f'[5/5] {work}: {npg} light pages of {a.page}px'
          + (f' ({nsplit} nodes: a page over {MAX_SURFACES} materials is split)'
             if nsplit > npg else '') + f', {len(mats)} materials, '
          f'{len(tex_files)} textures')
    if not a.install:
        print('[5/5] to pack it into the game: rerun with --install, or\n'
              f'      godot --headless --script "{work}/pack_zone_map.gd" -- "{work}" "<game>/custom_maps/{name}"')
        return
    from zonemap import locate
    root = locate.find_zone_custom_maps() if a.install == 'auto' else a.install
    dest = os.path.join(root, name)
    godot = find_godot(a.godot)
    if not godot:
        raise SystemExit('no Godot 4.4+ editor found: download it from godotengine.org and pass --godot <exe>')
    print(f'[5/5] packing with {godot}')
    r = subprocess.run([godot, '--headless', '--script', os.path.join(work, 'pack_zone_map.gd'), '--', work, dest],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in r.stdout.decode('utf-8', 'replace').splitlines():   # bytes: see "Three things the first real --cycles run found"
        if line.strip() and not line.startswith('Godot Engine'):
            print('      ' + line)
    if r.returncode:
        raise SystemExit('packing failed (above)')
    print(f'[5/5] installed -> {dest}  (beta: untick Legacy lighting, pick {name})')


def write_glb(a, g, imgs, pages, uvout, bake_prims):
    flat_n = None
    if a.flat_normals:
        flat_n, _ = parse_flat_normal(a.flat_normals)
    blend = float(getattr(a, 'normal_blend', 0.0) or 0.0)
    J = g.j
    out = Glb(generator='zone-map-forge bake_glb (lightmap baked into the albedo)')
    mat_map = copy_materials(g, out, a.roughness_min)

    flat = np.zeros((4, 4, 3), np.uint8)
    flat[:, :, 0] = 128; flat[:, :, 1] = 128; flat[:, :, 2] = 255
    nflat = out.image(_png(Image.fromarray(flat, 'RGB')), 'lm_flat_normal')
    atlas_mat = {}
    npng = 0
    for p in pages:
        name = f'{MAT_PREFIX}{p}'
        arr = imgs[p]
        masked = arr.shape[2] == 4
        if masked:
            # a cut-out cannot survive JPEG, and the loader picks
            # CustomMaterialAlpha off has_alpha(), so this page has to be PNG
            tex = out.image(_png(Image.fromarray(arr, 'RGBA')), name)
            npng += 1
        else:
            tex = out.image(_jpg(Image.fromarray(arr, 'RGB'), a.jpeg_q), name)
        # roughness 1: env_dynamic leaves the sky as the reflection source, so
        # anything smoother turns a dark baked albedo into a mirror of the sky
        # - the "glossy dark areas". There is no gloss worth keeping once the
        # light is in the texture.
        atlas_mat[p] = out.material(name, albedo_tex=tex, normal_tex=nflat,
                                    roughness=max(a.roughness_min, 0.04),
                                    metallic=0.0,
                                    alpha_mode='MASK' if masked else 'OPAQUE',
                                    double_sided=masked)
    if npng:
        print(f'[5/5] {npng} of {len(pages)} pages are cut-out (RGBA PNG)')

    prim_src = {(b[0], b[1]): b for b in bake_prims}
    mesh_map = {}
    for mi, m in enumerate(J['meshes']):
        prims = []
        for pi, pr in enumerate(m['primitives']):
            baked = uvout.get((mi, pi))
            if baked:
                _mi, _pi, P, N, UV, idx, matidx = prim_src[(mi, pi)]
                pos_l = g.acc(pr['attributes']['POSITION']).astype(np.float32)
                nrm_l = (g.acc(pr['attributes']['NORMAL']).astype(np.float32)
                         if 'NORMAL' in pr['attributes'] else None)
                by_page = collections.defaultdict(list)
                for k, (uv3, page) in baked.items():
                    by_page[page].append((k, uv3))
                for page, items in by_page.items():
                    tri = np.array([idx[k] for k, _ in items])
                    uv = np.concatenate([u for _, u in items]).astype(np.float32)
                    pp = pos_l[tri].reshape(-1, 3)
                    nn = (nrm_l[tri].reshape(-1, 3) if nrm_l is not None
                          else np.tile(np.array([0, 1, 0], np.float32), (len(pp), 1)))
                    if flat_n is not None:
                        if blend > 0.0:
                            # keep a trace of the real normal so every surface
                            # aims its specular lobe somewhere different
                            nn = (1.0 - blend) * flat_n[None, :] + blend * nn
                            ln = np.linalg.norm(nn, axis=1, keepdims=True)
                            nn = (nn / np.where(ln < 1e-6, 1.0, ln)
                                  ).astype(np.float32)
                        else:
                            nn = np.tile(flat_n,
                                         (len(pp), 1)).astype(np.float32)
                    ii = np.arange(len(pp), dtype=np.uint32).reshape(-1, 3)
                    prims.append((pp, nn, uv, ii, atlas_mat[page]))
                # triangles no chart claimed (degenerate) are dropped
            else:
                at = pr['attributes']
                pos = g.acc(at['POSITION']).astype(np.float32)
                nrm = (g.acc(at['NORMAL']).astype(np.float32) if 'NORMAL' in at
                       else np.tile(np.array([0, 1, 0], np.float32), (len(pos), 1)))
                uv = (g.acc(at['TEXCOORD_0']).astype(np.float32) if 'TEXCOORD_0' in at
                      else np.zeros((len(pos), 2), np.float32))
                idx = (g.acc(pr['indices']).astype(np.uint32).reshape(-1, 3)
                       if 'indices' in pr else
                       np.arange(len(pos), dtype=np.uint32).reshape(-1, 3))
                mat = mat_map.get(pr.get('material'), None)
                if mat is None:
                    mat = out.material(f'mesh{mi}_p{pi}', roughness=0.9)
                col = g.acc(at['COLOR_0']).astype(np.float32) if 'COLOR_0' in at else None
                prims.append((pos, nrm, uv, idx, mat, None, col))
        if prims:
            mesh_map[mi] = out.mesh(m.get('name', f'mesh{mi}'), prims)

    node_map = {}
    for ni, n in enumerate(J['nodes']):
        t = r = s = None
        if any(k in n for k in ('matrix', 'translation', 'rotation', 'scale')):
            t, r, s = decompose(node_matrix(n))
            if np.allclose(s, 1.0):
                s = None
            if np.allclose(r, [0, 0, 0, 1]):
                r = None
            if np.allclose(t, 0.0):
                t = None
        node_map[ni] = out.node(n.get('name', f'n{ni}'),
                                mesh=mesh_map.get(n.get('mesh')),
                                translation=t, rotation=r, scale=s)
        if n.get('extras'):
            out.j['nodes'][node_map[ni]]['extras'] = n['extras']
    for ni, n in enumerate(J['nodes']):
        kids = [node_map[c] for c in n.get('children', ()) if c in node_map]
        if kids:
            out.j['nodes'][node_map[ni]]['children'] = kids
    for rt in J['scenes'][J.get('scene', 0)]['nodes']:
        out.root(node_map[rt])

    # the same final pass forge.py makes: 'src_' keeps imported materials
    # clear of the game's z_ substitution table, and the game caches converted
    # materials by resource_name, so every name has to be unique
    from zonemap.convert import slug
    seen = {}
    for mm in out.j['materials']:
        if not mm['name'].startswith(('z_', 'src_', 'zone_')):
            mm['name'] = 'src_' + slug(mm['name'])
        if mm['name'] in seen:
            seen[mm['name']] += 1
            mm['name'] = f"{mm['name']}_{seen[mm['name']]}"
        else:
            seen[mm['name']] = 0

    dest = a.out or os.path.splitext(a.glb)[0] + '_baked.glb'
    n = out.save(dest)
    print(f'[5/5] {dest}  {n/1e6:.1f} MB  ({len(pages)} atlas pages)')
    if a.install:
        import shutil
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from zonemap import locate
        root = (locate.find_zone_custom_maps() if a.install == 'auto'
                else a.install)
        name = os.path.splitext(os.path.basename(dest))[0]
        folder = os.path.join(root, name)
        os.makedirs(folder, exist_ok=True)
        shutil.copy2(dest, os.path.join(folder, name + '.glb'))
        man = os.path.join(os.path.dirname(os.path.abspath(a.glb)), 'manifest.json')
        if os.path.isfile(man):
            out_man = os.path.join(folder, 'manifest.json')
            if a.sky_rotation:
                rot = (ZONE_BEST_SKY_ROTATION
                       if str(a.sky_rotation).strip().lower() == 'auto'
                       else float(a.sky_rotation))
                with open(man) as fh:
                    mj = json.load(fh)
                was = mj.get('skybox', {}).get('sky_rotation')
                for blk in ([mj.get('skybox')] +
                            [v.get('skybox') for v in
                             (mj.get('lighting_variations') or {}).values()]):
                    if isinstance(blk, dict):
                        blk['sky_rotation'] = rot
                with open(out_man, 'w') as fh:
                    json.dump(mj, fh, indent=1)
                print(f'[5/5] sky_rotation {was} -> {rot}')
            else:
                shutil.copy2(man, out_man)
        print(f'[5/5] installed -> {folder}{os.sep}{name}.glb')


if __name__ == '__main__':
    main()
