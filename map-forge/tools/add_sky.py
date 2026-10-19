#!/usr/bin/env python3
"""Put a map's sky brushes into a .glb already built, as forge.py now does.

forge.py keeps a Source map's sky brushes as solid zone_sky surfaces that
hide the rest of the map, with the game's sky drawn on them by the map_extras
mod (zonemap/sky.py says why). This adds them to a map built before that -
the forge's .glb, a bake folder's, or an installed map's - without
converting or baking again. They go in where the map has them: the .bsp's
sky faces scaled, moved and turned as convert_report.json says, those of the
3D skybox room left out at the same cut. Nothing else in the .glb changes.

    python tools/add_sky.py <map.glb or map folder> --bsp <map.bsp or .gma> [--map NAME]
                            [--report out/<map>/convert_report.json] [--glb-out PATH]

An installed map's download.json gets the new map.glb's checksum. A .glb
that already has zone_sky surfaces is left alone.
"""
import argparse, hashlib, json, math, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zonemap import bsp as B, build as BU, sky as SKY
from zonemap.convert import playable_centre, sky_camera, in_3d_skybox, SKYBOX_BIAS
from zonemap.glb import Glb


def load_bsp(spec, mapname=None):
    if spec.lower().endswith('.gma'):
        from zonemap.gma import Gma
        arc = Gma(spec)
        maps = [m for m, _ in arc.maps()]
        want = os.path.splitext(os.path.basename(mapname))[0].lower() if mapname else None
        pick = next((m for m in maps if want and os.path.splitext(os.path.basename(m))[0].lower() == want),
                    maps[0] if maps else None)
        if pick is None:
            raise SystemExit(f'no .bsp inside {spec}')
        return B.Bsp(pick, data=arc.read(pick))
    return B.Bsp(spec)


def read_glb(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'glTF':
            raise SystemExit(f'not a .glb: {path}')
        f.read(8)
        ln, _ = struct.unpack('<II', f.read(8))
        j = json.loads(f.read(ln))
        bin_at = 12 + 8 + ln
        h = f.read(8)
        bln = struct.unpack('<II', h)[0] if len(h) == 8 else 0
    return j, bin_at, bln


def sky_tris(bsp, rep):
    """the sky faces where this map has them: -> positions, normals"""
    scale = float(rep.get('scale_m_per_unit', BU.UNIT))
    shift = np.array(rep.get('origin_shift', [0, 0, 0]), np.float64)
    t = math.radians(float(rep.get('map_turn_deg') or 0.0))
    R = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    pos, nrm = SKY.faces(bsp, scale)
    n_all = len(pos) // 3
    sc, play = sky_camera(bsp), playable_centre(bsp)
    cut = float((rep.get('geometry_stats') or {}).get('skybox_cut') or SKYBOX_BIAS)
    if n_all and sc is not None and play is not None and len(play):
        sky_g = BU.to_gltf(sc[None, :], scale)[0]
        play_pts = BU.to_gltf(play, scale)
        cen = pos.reshape(-1, 3, 3).mean(axis=1)
        pos, nrm = SKY.keep_tris(pos, nrm, [not in_3d_skybox(c, sky_g, play_pts, cut) for c in cen])
    pos = ((pos.astype(np.float64) + shift) @ R.T).astype(np.float32)
    nrm = (nrm.astype(np.float64) @ R.T).astype(np.float32)
    return pos, nrm, n_all - len(pos) // 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('target', help='a .glb, or a folder with map.glb (a bake folder or an installed map)')
    ap.add_argument('--bsp', required=True, help='the map\'s .bsp, or the .gma it came in')
    ap.add_argument('--map', help='which map inside a .gma')
    ap.add_argument('--report', help='convert_report.json (default: beside the .glb)')
    ap.add_argument('--cell', type=float, default=48.0)
    ap.add_argument('--glb-out', help='write the new .glb here instead of over the old one')
    a = ap.parse_args()
    glb_path = os.path.join(a.target, 'map.glb') if os.path.isdir(a.target) else a.target
    rep_path = a.report or os.path.join(os.path.dirname(os.path.abspath(glb_path)), 'convert_report.json')
    if not os.path.isfile(rep_path):
        raise SystemExit(f'no {rep_path}: pass --report out/<map>/convert_report.json')
    rep = json.load(open(rep_path))
    j, bin_at, bln = read_glb(glb_path)
    if any(SKY.is_sky_node(n) for n in j['nodes']):
        print(f'{glb_path}: already has zone_sky surfaces - left alone')
        return
    pos, nrm, dropped = sky_tris(load_bsp(a.bsp, a.map), rep)
    if not len(pos):
        print(f'{glb_path}: the map has no sky faces - nothing to add')
        return
    # the new parts in a Glb of their own, then moved onto the end of this one
    g = Glb()
    nodes = SKY.add(g, pos, nrm, cell_m=a.cell)
    base = (bln + 3) // 4 * 4
    off = {k: len(j.get(k, [])) for k in ('bufferViews', 'accessors', 'meshes', 'materials', 'nodes')}
    for v in g.j['bufferViews']:
        v['byteOffset'] += base
        j.setdefault('bufferViews', []).append(v)
    for acc in g.j['accessors']:
        acc['bufferView'] += off['bufferViews']
        j.setdefault('accessors', []).append(acc)
    for m in g.j['materials']:
        j.setdefault('materials', []).append(m)
    for me in g.j['meshes']:
        for pr in me['primitives']:
            pr['attributes'] = {k: v + off['accessors'] for k, v in pr['attributes'].items()}
            pr['indices'] += off['accessors']
            pr['material'] += off['materials']
        j['meshes'].append(me)
    for n in g.j['nodes']:
        n['mesh'] += off['meshes']
        j['nodes'].append(n)
    j['nodes'].append({'name': 'sky_brushes', 'children': [i + off['nodes'] for i in nodes]})
    j['scenes'][j.get('scene', 0)]['nodes'].append(len(j['nodes']) - 1)
    used = j.setdefault('extensionsUsed', [])
    if 'KHR_materials_unlit' not in used:
        used.append('KHR_materials_unlit')
    add = bytes(g.blob) + b'\0' * ((4 - len(g.blob) % 4) % 4)
    new_bln = base + len(add)
    j['buffers'][0]['byteLength'] = new_bln
    js = json.dumps(j, separators=(',', ':')).encode('utf-8')
    js += b' ' * ((4 - len(js) % 4) % 4)
    dst = a.glb_out or glb_path + '.part'
    with open(glb_path, 'rb') as fi, open(dst, 'wb') as fo:
        fo.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + new_bln))
        fo.write(struct.pack('<II', len(js), 0x4E4F534A))
        fo.write(js)
        fo.write(struct.pack('<II', new_bln, 0x004E4942))
        fi.seek(bin_at + 8)
        left = bln
        while left:
            b = fi.read(min(left, 64 << 20))
            if not b:
                raise SystemExit('the .glb ended early')
            fo.write(b)
            left -= len(b)
        fo.write(b'\0' * (base - bln))
        fo.write(add)
    if not a.glb_out:
        os.replace(dst, glb_path)
    dj = os.path.join(os.path.dirname(os.path.abspath(glb_path)), 'download.json')
    if os.path.isfile(dj) and not a.glb_out:
        st = json.load(open(dj))
        if 'map.glb' in st.get('files', {}):
            h = hashlib.sha256()
            with open(glb_path, 'rb') as f:
                for b in iter(lambda: f.read(64 << 20), b''):
                    h.update(b)
            st['files']['map.glb'] = h.hexdigest()
            json.dump(st, open(dj, 'w'), indent='\t')
    print(f'{glb_path}: {len(pos) // 3} sky triangles ({SKY.area(pos):,.0f} m2) in {len(nodes)} zone_sky nodes'
          + (f' ({dropped} more were the 3D skybox\'s, left out)' if dropped else '')
          + (f' -> {a.glb_out}' if a.glb_out else ''))


if __name__ == '__main__':
    main()
