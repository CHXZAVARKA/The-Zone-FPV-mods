#!/usr/bin/env python3
"""Pack a --zone-v3 bake folder into the game, as bake_glb --install does.

    python tools/pack_v3.py out/<map>/<map>_v3_zone_v3 [--install DIR] [--godot EXE] [--gpu]

For a bake folder changed after its bake (tools/relayout_v3.py,
tools/split_v3_pages.py): it runs Godot 4.4+ (found in Downloads, Desktop or
Program Files, or --godot) on the folder's pack_zone_map.gd, into
<game>/custom_maps/<map>_v3 (or --install DIR/<map>_v3). When bake_info.json
says reuse_textures, the installed map's textures are kept and only the light
is packed again.

Godot squeezes the light into the game's format (BC6H) on the CPU when it runs
headless, as bake_glb --install runs it: about two minutes a page on two
slow cores. --gpu runs it with a window instead, so it uses the graphics card:
a few seconds for the lot, with the light about 1 % off on average where the
CPU is 0.2 % (measured on de_dust2's pages).
"""
import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bake', help='the bake folder (<map>_v3_zone_v3)')
    ap.add_argument('--install', default='auto', help="the game's custom_maps folder (default: found)")
    ap.add_argument('--godot', help='Godot 4.4+ editor executable')
    ap.add_argument('--gpu', action='store_true',
                    help='compress the light on the graphics card (a window opens): much faster, a little less exact')
    a = ap.parse_args()
    from bake_glb import find_godot
    from zonemap import locate
    work = os.path.abspath(a.bake).rstrip('/\\')
    name = os.path.basename(work)
    name = name[:-len('_zone_v3')] if name.endswith('_zone_v3') else name
    root = locate.find_zone_custom_maps() if a.install == 'auto' else a.install
    if not root:
        raise SystemExit("couldn't find the game's custom_maps folder: pass --install <folder>")
    dest = os.path.join(root, name)
    godot = find_godot(a.godot)
    if not godot:
        raise SystemExit('no Godot 4.4+ editor found: download it from godotengine.org and pass --godot <exe>')
    gd = os.path.join(work, 'pack_zone_map.gd')
    shutil.copy2(os.path.join(HERE, 'pack_zone_map.gd'), gd)
    print(f'packing {work}\n     -> {dest}\n   with {godot}')
    r = subprocess.run([godot] + ([] if a.gpu else ['--headless']) + ['--script', gd, '--', work, dest],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in r.stdout.decode('utf-8', 'replace').splitlines():
        if line.strip() and not line.startswith('Godot Engine'):
            print('   ' + line)
    if r.returncode:
        raise SystemExit('packing failed (above)')
    print(f'installed -> {dest}')


if __name__ == '__main__':
    main()
