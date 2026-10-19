#!/usr/bin/env python3
"""Put a map's turn into the map itself, once.

Maps forged before this change carry forge.py's turn (--recenter: the spawn
faces -Z) on the glTF's root node. This moves it into the map, as forge.py now
does when it builds one: world meshes get it in their vertices, props and
other placed nodes in their own transforms, and the root node keeps none. The
game shows the map exactly as before (every vertex is checked), bake_glb no
longer has a turned root to deal with, and in Blender the map is laid out as
you fly it.

    python tools/flatten_glb.py out/gm_br_pitfalls/gm_br_pitfalls.glb [more.glb ...]
    python tools/flatten_glb.py map.glb -o map_flat.glb

The file is rewritten in place unless -o is given. A map with no turned root
is left alone. convert_report.json beside it gets "turn_in_geometry": true;
its map_turn_deg (the turn from the .bsp's own layout) stays as it is, since
bake_glb still needs it to find the .bsp's light.
"""
import argparse, json, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zonemap import flatten as FL
from zonemap.glb import Glb

NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def load(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'glTF':
            raise SystemExit(f'not a .glb: {path}')
        f.read(8)
        j, b = None, bytearray()
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            ln, ty = struct.unpack('<II', h)
            d = f.read(ln)
            if ty == 0x4E4F534A:
                j = json.loads(d)
            elif ty == 0x004E4942:
                b = bytearray(d)
    return j, b


def world_boxes(j, b):
    """each mesh primitive's world-space vertices, by (node, primitive)"""
    out = {}

    def walk(i, M):
        n = j['nodes'][i]
        Mi = M @ FL.node_matrix(n)
        if 'mesh' in n:
            for pi, p in enumerate(j['meshes'][n['mesh']].get('primitives', [])):
                ai = p.get('attributes', {}).get('POSITION')
                if ai is None:
                    continue
                a = j['accessors'][ai]
                if a.get('componentType') != 5126 or a.get('bufferView') is None:
                    continue
                v = j['bufferViews'][a['bufferView']]
                P = np.ndarray((a['count'], 3), '<f4', buffer=b,
                               offset=v.get('byteOffset', 0) + a.get('byteOffset', 0),
                               strides=(v.get('byteStride') or 12, 4)).astype(np.float64)
                out[(i, pi)] = P @ Mi[:3, :3].T + Mi[:3, 3]
        for c in n.get('children', ()):
            walk(c, Mi)

    for r in j['scenes'][j.get('scene', 0)]['nodes']:
        walk(r, np.eye(4))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('glb', nargs='+')
    ap.add_argument('-o', '--out', help='write here instead (one input only)')
    a = ap.parse_args()
    if a.out and len(a.glb) > 1:
        raise SystemExit('-o takes one input')
    for path in a.glb:
        t0 = time.time()
        j, b = load(path)
        if not FL.root_turns(j):
            print(f'{path}: no turned root node - left as it is')
            continue
        before = world_boxes(j, b)
        st = FL.flatten_roots(j, b)
        after = world_boxes(j, b)
        worst = max((float(np.abs(before[k] - after[k]).max()) for k in before
                     if k in after and len(before[k])), default=0.0)
        size = max((float(np.abs(v).max()) for v in before.values() if len(v)), default=1.0)
        if st['refused'] or before.keys() != after.keys() or worst > max(1e-3, size * 1e-6):
            raise SystemExit(f'{path}: NOT written - {FL.describe(st)}; '
                             f'vertices moved up to {worst:.3g} m')
        g = Glb()
        g.j, g.blob = j, b
        dst = a.out or path
        tmp = dst + '.part'
        g.save(tmp)
        os.replace(tmp, dst)
        rep = os.path.join(os.path.dirname(os.path.abspath(path)), 'convert_report.json')
        if not a.out and os.path.isfile(rep):
            r = json.load(open(rep))
            r['turn_in_geometry'] = True
            json.dump(r, open(rep, 'w'), indent=1)
        print(f'{path}: {FL.describe(st)}; every vertex where it was '
              f'(within {worst:.1g} m) -> {dst} in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
