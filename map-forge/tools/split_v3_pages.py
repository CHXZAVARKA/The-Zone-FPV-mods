#!/usr/bin/env python3
"""Split light pages that have more than 256 materials, in a map already baked.

Godot gives a mesh at most 256 surfaces (one per material) and drops the rest
when it loads the .glb, with nothing on screen but a line in the log for each:

    ERROR: Condition "surfaces.size() == RenderingServerEnums::MAX_MESH_SURFACES" is true.

bake_glb --zone-v3 used to put everything charted onto a light page into one
node, one surface per material, so a busy page lost every material past the
256th - gm_br_pitfalls lost 1,435, door slabs among them (the knob and the
glass, on other pages, still showed). bake_glb now writes such a page as
several nodes sharing its lightmap. This does the same to a map baked before,
without baking again: the light, UVs and textures are untouched, only the
.glb's node list and the lightmap index change.

    python tools/split_v3_pages.py out/<map>/<map>_v3_zone_v3     # the bake folder
    python tools/split_v3_pages.py "<game>/custom_maps/<map>_v3"   # the installed map

On the bake folder it adds the new nodes to bake_info.json, for
pack_zone_map.gd; on an installed map it adds them to materials_<tier>.json
and download.json, so the game picks it up as it is - no packing needed.
"""
import argparse, hashlib, json, os, re, shutil, struct, sys

MAX_SURFACES = 256
NAME = re.compile(r'^(lm\d+)_zlms-(\d+)$')


def read_glb(path):
    with open(path, 'rb') as f:
        head = f.read(12)
        if head[:4] != b'glTF':
            raise SystemExit(f'not a .glb: {path}')
        ln, ty = struct.unpack('<II', f.read(8))
        j = json.loads(f.read(ln))
        bin_at = 12 + 8 + ln
        f.seek(bin_at)
        h = f.read(8)
        bln = struct.unpack('<II', h)[0] if len(h) == 8 else 0
    return j, bin_at, bln


def write_glb(src, j, bin_at, bln, dst):
    js = json.dumps(j, separators=(',', ':')).encode('utf-8')
    js += b' ' * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + (8 + bln if bln else 0)
    with open(src, 'rb') as fi, open(dst, 'wb') as fo:
        fo.write(struct.pack('<III', 0x46546C67, 2, total))
        fo.write(struct.pack('<II', len(js), 0x4E4F534A))
        fo.write(js)
        if bln:
            fi.seek(bin_at)
            left = 8 + bln
            while left:
                b = fi.read(min(left, 64 << 20))
                if not b:
                    raise SystemExit('the .glb ended early')
                fo.write(b)
                left -= len(b)


def split(j):
    """-> {new node name: the page it shares}; j is changed in place"""
    added = {}
    roots = j['scenes'][j.get('scene', 0)]['nodes']
    parent = {c: i for i, n in enumerate(j['nodes']) for c in n.get('children', ())}
    for ni in range(len(j['nodes'])):
        n = j['nodes'][ni]
        m = NAME.match(n.get('name', ''))
        if not m or 'mesh' not in n:
            continue
        prims = j['meshes'][n['mesh']]['primitives']
        if len(prims) <= MAX_SURFACES:
            continue
        parts = -(-len(prims) // MAX_SURFACES)
        size = -(-len(prims) // parts)                  # as even as it goes
        chunks = [prims[k:k + size] for k in range(0, len(prims), size)]
        j['meshes'][n['mesh']]['primitives'] = chunks[0]
        for k, ch in enumerate(chunks[1:], 1):
            name = f'{m.group(1)}s{k}_zlms-{m.group(2)}'
            j['meshes'].append({'name': name, 'primitives': ch})
            nd = {kk: vv for kk, vv in n.items() if kk not in ('mesh', 'children', 'name')}
            nd.update(name=name, mesh=len(j['meshes']) - 1)
            j['nodes'].append(nd)
            new = len(j['nodes']) - 1
            if ni in parent:
                j['nodes'][parent[ni]]['children'].append(new)
            else:
                roots.append(new)
            added[name] = n['name']
        print(f"  {n['name']}: {len(prims)} materials -> {len(chunks)} nodes of up to {size}")
    return added


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(64 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('folder', help='a bake folder (<map>_v3_zone_v3) or an installed map folder')
    ap.add_argument('--glb-out', help='write the new map.glb here instead of over the old one '
                                      '(then copy it in yourself)')
    a = ap.parse_args()
    d = a.folder
    glb = os.path.join(d, 'map.glb')
    j, bin_at, bln = read_glb(glb)
    added = split(j)
    over = [n['name'] for n in j['nodes'] if 'mesh' in n
            and len(j['meshes'][n['mesh']]['primitives']) > MAX_SURFACES]
    if over:
        print(f'  still over {MAX_SURFACES} surfaces, not light pages (left as they are): {over[:5]}')
    if not added:
        print(f'{d}: no light page has more than {MAX_SURFACES} materials - nothing to do')
        return
    changed = []
    bi = os.path.join(d, 'bake_info.json')
    if os.path.isfile(bi):
        info = json.load(open(bi))
        have = {o['name'] for o in info['objects']}
        size = {o['name']: o['size'] for o in info['objects']}
        for new, page in added.items():
            if new not in have:
                info['objects'].append({'name': new, 'size': size[page], 'lightmap': page})
        json.dump(info, open(bi, 'w'), indent=1)
        print(f'  bake_info.json: {len(added)} objects sharing their page\'s lightmap '
              f'(pack it with the pack_zone_map.gd from tools/, which reads that)')
        shutil.copy2(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pack_zone_map.gd'),
                     os.path.join(d, 'pack_zone_map.gd'))
    for tier in ('high', 'medium', 'low', 'potato'):
        p = os.path.join(d, f'materials_{tier}.json')
        if not os.path.isfile(p):
            continue
        doc = json.load(open(p))
        for var, entries in doc.get('lightmaps', {}).items():
            for new, page in added.items():
                if page in entries:
                    entries[new] = entries[page]
        for var, means in doc.get('lightmap_means', {}).items():
            for new, page in added.items():
                if page in means:
                    means[new] = means[page]
        with open(p, 'w') as fh:
            fh.write(json.dumps(doc, indent=' '))
        changed.append(os.path.basename(p))
    dst = a.glb_out or glb + '.part'
    write_glb(glb, j, bin_at, bln, dst)
    if not a.glb_out:
        os.replace(dst, glb)
        changed.append('map.glb')
    dj = os.path.join(d, 'download.json')
    if os.path.isfile(dj) and not a.glb_out:
        st = json.load(open(dj))
        for fn in changed:
            if fn in st.get('files', {}):
                st['files'][fn] = sha256(os.path.join(d, fn))
        json.dump(st, open(dj, 'w'), indent='\t')
    others = [c for c in changed if c != 'map.glb']
    print(f'{d}: {len(added)} nodes added; ' + (f'map.glb written to {a.glb_out}' if a.glb_out
                                                 else 'map.glb rewritten') +
          (f'; {", ".join(others)} updated' if others else ''))


if __name__ == '__main__':
    main()
