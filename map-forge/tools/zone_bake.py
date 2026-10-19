"""zone_bake.py - turn a Blender map into a baked-light map for The Zone FPV (the beta's
lighting v3: real lightmaps, the official maps' camera). Everything runs on your PC.

ONE-TIME SETUP
  * Godot 4.4 or newer, the plain editor (godotengine.org > Download, Windows): the packer
    uses its BC6H/BC7 compressors. Set GODOT_EXE below, or leave it '' and the script looks
    in Downloads for Godot_v4.*_win64*_console.exe.
  * GAME_DIR below: the game's folder (the Steam one is found on its own).

FOR EACH MAP
  1. File > Open your map's .blend (not a new file).
  2. Scripting tab > Text > Open > zone_bake.py > Run Script. (Copy this file anywhere.)
     Window > Toggle System Console shows progress. Blender stops responding until it is done.
     Or without the window:  blender.exe -b "my map.blend" --python zone_bake.py
  3. That's it: it bakes, denoises, packs and installs <GAME_DIR>/custom_maps/<MAP_NAME>.
     In the beta, untick Settings > Legacy lighting and pick the map from the custom maps.
  To pack again without baking (e.g. after changing the game folder): double-click
  zone_bake/pack.bat.

WHAT IT DOES (in a copy, zone_bake/<name>_zone.blend - your .blend is left as it was)
  * Array modifiers become one object per copy; very heavy meshes are decimated (COLUMN_TRIS).
  * Single-layer meshes become two-sided (Blender draws both sides of a face, the game only
    the front); closed meshes get their faces pointing out.
  * Materials become the game's: image textures carry over, a Brick Texture becomes a seamless
    tile image, other procedural colours become flat colours. Glass, transparent and alpha
    materials become see-through (SEE_THROUGH_*).
  * Every object gets a lightmap UV map, sized by area (DENSITY), and light portals go in
    openings to the sky.
  * Cycles bakes each object's light (diffuse, direct + indirect, without colour) with what
    renders in the file - its World and its lights - and Blender's denoiser cleans it up.
  * zone_bake/ gets map.glb, occluder.bin, lightmaps/*.exr, textures/*.png, bake_info.json,
    pack_zone_map.gd and pack.bat; then Godot packs the map folder into the game.

Only you, and whoever you give the map folder to, see the baked light: the server only has
what was uploaded. Don't upload from this folder - its map.glb has no textures in it.
"""
import bpy, bmesh, math, os, json, time, re
import mathutils
import numpy as np

# ---------------------------------------------------------------- settings
MAP_NAME = ''           # the map's folder name in custom_maps; '' = the .blend's name
GAME_DIR = ''           # the game's folder; '' = Steam's usual place
GODOT_EXE = ''          # a Godot 4.4+ editor .exe; '' = look in Downloads
AUTO_PACK = True        # pack and install into the game when the bake is done
DENOISE = True          # clean up the bake with Blender's denoiser (OpenImageDenoise)
SEE_THROUGH_ALPHA = 0.8 # glass/water: 0 = invisible, 1 = solid
SEE_THROUGH_TINT = 0.3  # glass/water colour: its Blender colour times this
SEE_THROUGH_SOLID = False   # True = you hit glass/water; False = you fly through it
DENSITY = 16            # lightmap texels per metre (16 = about 6 cm, before packing)
SAMPLES = 512           # Cycles samples per lightmap texel
MAX_SIZE = 2048         # largest lightmap (the game takes 64 to 4096)
MIN_SIZE = 64
COLUMN_TRIS = 4000      # a mesh copy heavier than this is decimated to about this
INSTANCE_TRIS = 300000  # budget for everything made of instances (geometry-nodes ceilings and
                        #   scatters, collection instances): decimated evenly to fit
INSTANCE_CELL = 8.0     # instances are merged into one lightmapped piece per cell this size (m)
DIFFUSE_BOUNCES = 6
VARIATION = 'day'       # the lighting variation's name in the game
SKYBOX = 'cloudy_sunny' # the game's sky seen through openings: cloudy_sunny, sunny,
SKY_ROTATION = 0.0      #   dawn, sunset or industrial_sunset
SKY_ENERGY = 1.0
PORTALS = True          # light portals in openings that face the sky
GPU = True              # use the GPU set in Preferences > System, if there is one
TWO_SIDED_MAX_FACES = 3000  # meshes up to this many faces are made to show both sides as
                        #   Blender does (heavier ones, e.g. detailed props, are left as they are)

if os.environ.get('ZONE_BAKE_TEST'):        # quick check of the whole pipeline
    DENSITY, SAMPLES, MIN_SIZE = 3, 4, 64

LIGHTMAP_SIZES = [64, 128, 256, 512, 1024, 2048, 3072, 4096]
# the packer, written to zone_bake/pack_zone_map.gd and run with Godot
PACKER_GD = r'''extends SceneTree
## Packs a zone_bake folder into the map folder the 2026 beta loads with real lightmaps.
##   godot --headless --script pack_zone_map.gd -- "<zone_bake folder>" "<game>/custom_maps/<name>"
## Needs a Godot 4.4+ editor build (the plain editor .exe; export templates can't compress) (it compresses with Godot's own BC6H/BC7 encoders, which is
## what the game's loader checks: BC6H unsigned, full mips, exactly <size> x the tier's scale).

const TIERS := [
	{"name": "high", "texture_max": 4096, "lightmap_scale": 1.0},
	{"name": "medium", "texture_max": 2048, "lightmap_scale": 0.5},
	{"name": "low", "texture_max": 1024, "lightmap_scale": 0.25},
	{"name": "potato", "texture_max": 512, "lightmap_scale": 0.125},
]


func _init() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() < 2:
		printerr("usage: godot --headless --script pack_zone_map.gd -- <zone_bake folder> <map folder>")
		quit(2)
		return
	var bake: String = args[0].replace("\\", "/").trim_suffix("/")
	var out: String = args[1].replace("\\", "/").trim_suffix("/")
	var info = JSON.parse_string(FileAccess.get_file_as_string(bake + "/bake_info.json"))
	if not (info is Dictionary):
		printerr("no bake_info.json in " + bake + " - run zone_bake.py in Blender first")
		quit(1)
		return
	DirAccess.make_dir_recursive_absolute(out)
	var variation: String = info.get("variation", "day")
	var t0 := Time.get_ticks_msec()
	# textures: one id per image file
	var tex_ids := {}
	for mname in info.materials:
		var m: Dictionary = info.materials[mname]
		if m.get("albedo") and not tex_ids.has(m.albedo):
			tex_ids[m.albedo] = "t%d" % tex_ids.size()
	var files := ["map.glb", "occluder.bin", "manifest.json"]
	var dds_cache := {}
	for tier in TIERS:
		var lms := {}
		var means := {}
		var lf := FileAccess.open(out + "/lightmaps_%s.blob" % tier.name, FileAccess.WRITE)
		for ob in info.objects:
			var img := Image.load_from_file(bake + "/lightmaps/" + ob.name + ".exr")
			if img == null:
				fail("can't read lightmaps/%s.exr" % ob.name)
				return
			img.convert(Image.FORMAT_RGBH)
			var size := int(ob.size * float(tier.lightmap_scale))
			while img.get_width() > size:       # halve, so each texel is the average of four
				img.resize(img.get_width() / 2, img.get_height() / 2, Image.INTERPOLATE_BILINEAR)
			if img.get_width() != size or img.get_height() != size:
				fail("%s is %s, expected %dx%d" % [ob.name, img.get_size(), ob.size, ob.size])
				return
			means[ob.name] = _mean(img)
			img.generate_mipmaps()
			img.compress(Image.COMPRESS_BPTC, Image.COMPRESS_SOURCE_GENERIC)
			if img.get_format() != Image.FORMAT_BPTC_RGBFU:
				fail("%s: compressed to format %d, not BC6H unsigned (negative light?)" % [ob.name, img.get_format()])
				return
			var dds := _dds(img, 95)        # DXGI_FORMAT_BC6H_UF16
			lms[ob.name] = {"offset": lf.get_position(), "length": dds.size()}
			lf.store_buffer(dds)
		lf.close()
		var texs := {}
		var tf := FileAccess.open(out + "/textures_%s.blob" % tier.name, FileAccess.WRITE)
		for path in tex_ids:
			var img := Image.load_from_file(bake + "/" + path)
			if img == null:
				fail("can't read " + path)
				return
			var mx := int(tier.texture_max)
			var w := img.get_width()
			var h := img.get_height()
			while w > mx or h > mx:
				w = maxi(4, w / 2)
				h = maxi(4, h / 2)
			var key := "%s@%dx%d" % [path, w, h]
			if not dds_cache.has(key):        # most textures are the same size in every tier
				img.convert(Image.FORMAT_RGBA8)
				while img.get_width() > w:
					img.resize(maxi(4, img.get_width() / 2), maxi(4, img.get_height() / 2), Image.INTERPOLATE_BILINEAR)
				img.generate_mipmaps()
				img.compress(Image.COMPRESS_BPTC, Image.COMPRESS_SOURCE_SRGB)
				dds_cache[key] = _dds(img, 98)        # DXGI_FORMAT_BC7_UNORM
			var dds: PackedByteArray = dds_cache[key]
			texs[tex_ids[path]] = {"offset": tf.get_position(), "length": dds.size()}
			tf.store_buffer(dds)
		tf.close()
		var mats := {}
		for mname in info.materials:
			var m: Dictionary = info.materials[mname]
			var e := {"color": m.color, "roughness": m.roughness, "metallic": m.metallic,
				"alpha": m.alpha, "transparent": m.alpha}
			if m.get("albedo"):
				e["albedo"] = tex_ids[m.albedo]
			if float(m.get("emission_energy", 0.0)) > 0.0:
				e["emission_color"] = m.emission_color
				e["emission_energy"] = m.emission_energy
			mats[mname] = e
		var doc := {"version": 1, "tier": tier.name, "lightmap_scale": tier.lightmap_scale,
			"textures": texs, "materials": mats, "lightmaps": {variation: lms},
			"lightmap_means": {variation: means}}
		_write(out + "/materials_%s.json" % tier.name, JSON.stringify(doc, " "))
		files += ["textures_%s.blob" % tier.name, "lightmaps_%s.blob" % tier.name, "materials_%s.json" % tier.name]
		print("%-6s %3d lightmaps  %5.1f MB   (%d s)" % [tier.name, lms.size(),
			FileAccess.get_file_as_bytes(out + "/lightmaps_%s.blob" % tier.name).size() / 1e6,
			(Time.get_ticks_msec() - t0) / 1000])
	DirAccess.copy_absolute(bake + "/map.glb", out + "/map.glb")
	DirAccess.copy_absolute(bake + "/occluder.bin", out + "/occluder.bin")
	var sky: Dictionary = info.lighting_variation.skybox
	var manifest := {"baked": true, "lighting_variations": {variation: info.lighting_variation},
		"skybox": {"skybox_name": sky.skybox_name, "sky_rotation": sky.sky_rotation, "energy": sky.energy},
		"spawn_point": info.spawn_point, "assets": []}
	_write(out + "/manifest.json", JSON.stringify(manifest, " "))
	# the game counts a quality tier as there only if download.json lists all its files
	var state := {"version": "local-" + Time.get_datetime_string_from_system().replace(":", ""), "files": {}}
	for f in files:
		state.files[f] = FileAccess.get_sha256(out + "/" + f)
	_write(out + "/download.json", JSON.stringify(state, "\t"))
	print("done in %d s: %s" % [(Time.get_ticks_msec() - t0) / 1000, out])
	quit(0)


func _mean(img: Image) -> float:
	# the image's average light, from a small copy
	var small := img.duplicate() as Image
	while small.get_width() > 8:
		small.resize(small.get_width() / 2, small.get_height() / 2, Image.INTERPOLATE_BILINEAR)
	var s := 0.0
	for y in small.get_height():
		for x in small.get_width():
			s += small.get_pixel(x, y).get_luminance()
	return s / (small.get_width() * small.get_height())


func _dds(img: Image, dxgi: int) -> PackedByteArray:
	## a DDS file with a DX10 header, as the game's DDS reader takes it (Godot 4.4 has no
	## save_dds_to_buffer): magic, 124-byte header, 20-byte DX10 header, then every mip
	var w := img.get_width()
	var h := img.get_height()
	var b := StreamPeerBuffer.new()
	b.put_data("DDS ".to_ascii_buffer())
	for v in [124, 0xA1007, h, w, maxi(1, (w + 3) / 4) * maxi(1, (h + 3) / 4) * 16, 0, img.get_mipmap_count() + 1]:
		b.put_u32(v)
	for i in 11:
		b.put_u32(0)
	for v in [32, 0x4]:                       # pixel format: FourCC
		b.put_u32(v)
	b.put_data("DX10".to_ascii_buffer())
	for i in 5:
		b.put_u32(0)
	for v in [0x401008, 0, 0, 0, 0]:          # caps: texture, mipmaps, complex
		b.put_u32(v)
	for v in [dxgi, 3, 0, 1, 0]:              # DX10: format, 2D texture, array size 1
		b.put_u32(v)
	b.put_data(img.get_data())
	return b.data_array


func _write(path: String, text: String) -> void:
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string(text)
	f.close()


func fail(msg: String) -> void:
	printerr("pack_zone_map: " + msg)
	quit(1)
'''
# names the game would take from its own material library instead (substring match)
GAME_LIBRARY_NAMES = ['no_texture', 'Glass', 'Material.001', 'Concrete02', 'PavingTiles',
                      'GroundPebbles', 'PlasticABS', 'Poliigon_', 'TilesCeramicSquare']
PACKING = 0.62          # share of a lightmap the UV islands cover


def log(*a):
    print('[zone_bake]', *a, flush=True)


def slug(s):
    s = re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')
    return s or 'x'


def srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


# ---------------------------------------------------------------- setup
blend = bpy.data.filepath
scene = bpy.context.scene
view_layer = bpy.context.view_layer


def rendering_objects(lc=None, ok=True):
    """names of the objects in collections that render (not excluded, not hidden from renders)"""
    lc = lc or view_layer.layer_collection
    ok = ok and not lc.exclude and not getattr(lc.collection, 'hide_render', False)
    names = {o.name for o in lc.collection.objects} if ok else set()
    for ch in lc.children:
        names |= rendering_objects(ch, ok)
    return names


RENDERS = rendering_objects()


def renders(o):
    return o.name in RENDERS and not o.hide_render


to_bake = [o for o in scene.objects if renders(o) and (o.type == 'MESH' and len(o.data.polygons)
           or any(m.type == 'NODES' for m in getattr(o, 'modifiers', [])) or o.instance_type != 'NONE')]
if not to_bake:
    raise RuntimeError('Nothing to bake in %s: no meshes that render. Open your map first '
                       '(File > Open > the map\'s .blend), then run this script again.'
                       % (os.path.basename(blend) or 'this new, unsaved file'))
if not blend:
    raise RuntimeError('Save the .blend first (File > Save), then run this script again.')
if bpy.path.basename(blend).endswith('_zone.blend'):
    raise RuntimeError('This is a copy zone_bake made. Open the map\'s own .blend and run the script there.')
log('%s: %d objects that render' % (os.path.basename(blend), len(to_bake)))
SRC_DIR = os.path.dirname(blend)
STEM = os.path.splitext(os.path.basename(blend))[0]
NAME = MAP_NAME or slug(STEM)
OUT = os.path.join(SRC_DIR, 'zone_bake')
for d in ('', 'lightmaps', 'textures'):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
t_start = time.time()
if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, STEM + '_zone.blend'))
log('working in', bpy.data.filepath)
# the viewport shows what renders (the script reads the scene through the viewport's eyes)
for o in scene.objects:
    if renders(o):
        o.hide_viewport = False
        o.hide_set(False)
        for m in getattr(o, 'modifiers', []):
            m.show_viewport = m.show_render


def _show(lc):
    if not lc.exclude and not getattr(lc.collection, 'hide_render', False):
        lc.hide_viewport = False
        lc.collection.hide_viewport = False
        for ch in lc.children:
            _show(ch)


_show(view_layer.layer_collection)
view_layer.update()
scene = bpy.context.scene
view_layer = bpy.context.view_layer
scene.render.engine = 'CYCLES'


def setup_device():
    if not GPU:
        scene.cycles.device = 'CPU'
        return 'CPU'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        kinds = [prefs.compute_device_type] if prefs.compute_device_type != 'NONE' else []
        kinds += ['OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL']
        for k in kinds:
            try:
                prefs.compute_device_type = k
            except TypeError:
                continue
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == k]
            if gpus:
                for d in gpus:
                    d.use = True
                scene.cycles.device = 'GPU'
                return '%s: %s' % (k, ', '.join(d.name for d in gpus))
    except Exception as e:
        log('no GPU:', e)
    scene.cycles.device = 'CPU'
    return 'CPU'


# ---------------------------------------------------------------- materials
_outer = {}     # node group tree name -> the group node using it


def flat_nodes(tree, outer=None, depth=0):
    """every node of a material, including those inside node groups"""
    if outer is not None:
        _outer[tree.name] = outer
    out = []
    for n in tree.nodes:
        out.append(n)
        if n.type == 'GROUP' and n.node_tree and depth < 6:
            out += flat_nodes(n.node_tree, n, depth + 1)
    return out


def resolve(sock):
    """a socket fed by a node group's Group Input: the group node's own input instead"""
    for _ in range(6):
        if not sock.is_linked or sock.links[0].from_node.type != 'GROUP_INPUT':
            return sock
        l = sock.links[0]
        outer = _outer.get(l.from_node.id_data.name)
        nxt = outer and next((i for i in outer.inputs if i.identifier == l.from_socket.identifier), None)
        if nxt is None:
            return sock
        sock = nxt
    return sock


def linked_from(sock):
    sock = resolve(sock)
    return sock.links[0].from_node if sock.is_linked else None


def sock_value(sock):
    return resolve(sock).default_value


def mapping_scale(node):
    """the UV scale a Mapping node in front of a texture applies"""
    vec = node.inputs.get('Vector')
    m = linked_from(vec) if vec else None
    while m is not None and m.type == 'REROUTE':
        m = linked_from(m.inputs[0])
    if m is not None and m.type == 'MAPPING':
        s = m.inputs['Scale'].default_value
        return (s[0], s[1])
    return (1.0, 1.0)


def brick_tile(node, bricks=16, px=64):
    """a seamless image of a Brick Texture: `bricks` x `bricks` bricks, random tints"""
    i = {s.name: s for s in node.inputs}
    c1 = np.array(i['Color1'].default_value[:3]); c2 = np.array(i['Color2'].default_value[:3])
    cm = np.array(i['Mortar'].default_value[:3])
    msize = i['Mortar Size'].default_value; msmooth = i['Mortar Smooth'].default_value
    bias = i['Bias'].default_value; bw = i['Brick Width'].default_value; rh = i['Row Height'].default_value
    off_amt = node.offset; off_f = max(1, node.offset_frequency)
    rows = bricks if bricks % off_f == 0 else bricks * off_f
    n = bricks * px
    rng = np.random.default_rng(7)
    tint_tab = np.clip(rng.random((rows, bricks)) + bias, 0, 1)
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    px_ = (xs + 0.5) / n * bricks * bw
    py_ = (ys + 0.5) / n * rows * rh
    row = np.floor(py_ / rh).astype(int)
    offset = np.where(row % off_f, 0.0, bw * off_amt)
    num = np.floor((px_ + offset) / bw).astype(int)
    x = px_ + offset - bw * num
    y = py_ - rh * row
    tint = tint_tab[row % rows, num % bricks][..., None]
    mind = np.minimum(np.minimum(x, y), np.minimum(bw - x, rh - y))
    if msmooth == 0:
        mort = (mind < msize).astype(float)
    else:
        d = 1.0 - mind / msize
        t = np.clip(d / msmooth, 0, 1)
        mort = np.where(mind >= msize, 0.0, np.where(d < msmooth, t * t * (3 - 2 * t), 1.0))
    mort = mort[..., None]
    col = (c1 * (1 - tint) + c2 * tint) * (1 - mort) + cm * mort
    scale = node.inputs['Scale'].default_value
    # UV -> one tile of the image
    uv_scale = (scale / (bricks * bw), scale / (rows * rh))
    return col, uv_scale, n


def save_png(name, rgb_linear, alpha=None):
    h, w = rgb_linear.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=alpha is not None)
    px = np.ones((h, w, 4), np.float32)
    px[..., :3] = srgb(rgb_linear)          # stored as sRGB bytes in the PNG
    if alpha is not None:
        px[..., 3] = alpha
    img.colorspace_settings.name = 'sRGB'
    img.pixels.foreach_set(np.flipud(px).ravel())   # Blender rows go bottom-up
    path = os.path.join(OUT, 'textures', name + '.png')
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    return img, path


def upstream(node, want, depth=8):
    """the first node feeding `node` (inputs, then their inputs...) that `want` accepts"""
    seen, todo = set(), [(node, 0)]
    while todo:
        n, d = todo.pop(0)
        if n.name in seen or d > depth:
            continue
        seen.add(n.name)
        if want(n):
            return n
        for i in n.inputs:
            for l in i.links:
                todo.append((l.from_node, d + 1))
    return None


def as_colour(v):
    try:
        v = list(v)
    except TypeError:                       # a float socket
        return [float(v), float(v), float(v), 1.0]
    v = [float(x) for x in v][:4]
    return v + [1.0] * (4 - len(v)) if len(v) >= 3 else [0.8, 0.8, 0.8, 1.0]


def blackbody(k):
    """linear RGB of a black body at k kelvin, normalised to its brightest channel"""
    t = max(1000.0, min(40000.0, k)) / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    c = [max(0.0, min(255.0, x)) / 255.0 for x in (r, g, b)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]   # sRGB -> linear
    m = max(c) or 1.0
    return [x / m for x in c] + [1.0]


def guess_colour(node):
    """a flat colour for a procedural base colour: a Color Ramp's average, else the
    first colour input set on the node or above it, else light grey"""
    bb = upstream(node, lambda n: n.type == 'BLACKBODY')
    if bb:
        return blackbody(float(sock_value(bb.inputs['Temperature'])))
    ramp = upstream(node, lambda n: n.type == 'VALTORGB')
    if ramp:
        cs = [list(e.color) for e in ramp.color_ramp.elements]
        return [sum(c[i] for c in cs) / len(cs) for i in range(3)] + [1.0]
    def has_colour(n):
        return any(i.type == 'RGBA' and not i.is_linked for i in n.inputs)
    n = upstream(node, has_colour)
    if n:
        return as_colour(next(i for i in n.inputs if i.type == 'RGBA' and not i.is_linked).default_value)
    return [0.8, 0.8, 0.8, 1.0]


def find_emission(nodes):
    """(colour, strength) of the brightest Emission shader or Principled emission, or None"""
    best = None
    for n in nodes:
        if n.type == 'EMISSION':
            c, st = n.inputs['Color'], n.inputs['Strength']
        elif n.type == 'BSDF_PRINCIPLED' and 'Emission Strength' in n.inputs:
            c, st = n.inputs['Emission Color'], n.inputs['Emission Strength']
        else:
            continue
        c, st = resolve(c), resolve(st)
        strength = 1.0 if st.is_linked else float(st.default_value)
        col = guess_colour(linked_from(c)) if c.is_linked else as_colour(c.default_value)
        if strength > 0 and max(col[:3]) > 0 and (best is None or strength * max(col[:3]) > best[1] * max(best[0][:3])):
            best = (col, strength)
    return best


def analyse_material(mat):
    """what the game can show of a Blender material"""
    info = {'color': [0.8, 0.8, 0.8, 1.0], 'roughness': 0.6, 'metallic': 0.0, 'alpha': False,
            'albedo': None, 'uv_scale': (1.0, 1.0), 'note': '', 'emission': None}
    if mat is None or not mat.node_tree:
        return info
    nodes = flat_nodes(mat.node_tree)
    info['emission'] = find_emission(nodes)
    types = {n.type for n in nodes}
    glass = next((n for n in nodes if n.type == 'BSDF_GLASS'), None)
    if (glass or 'BSDF_TRANSPARENT' in types) and info['emission']:
        # a lamp's lens: it lights the room, so it stays solid and in the bake
        info['color'] = list(info['emission'][0])
        info['note'] = 'glowing glass (a lamp lens): solid and lit'
        return info
    if glass or 'BSDF_TRANSPARENT' in types:
        c = as_colour(sock_value(glass.inputs['Color'])) if glass else (0.6, 0.8, 0.8, 1)
        t = SEE_THROUGH_TINT
        info.update(color=[c[0] * t, c[1] * t, c[2] * t, SEE_THROUGH_ALPHA], roughness=0.05, alpha=True,
                    note='see-through (glass/transparent)' + ('' if SEE_THROUGH_SOLID else ', no collision'))
        return info
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None) or \
        next((n for n in nodes if n.type == 'BSDF_DIFFUSE'), None)
    if bsdf is None:
        if info['emission']:
            info['color'] = list(info['emission'][0])
            info['note'] = 'emission only: glows in its colour'
        else:
            info['note'] = 'no Principled/Diffuse BSDF: grey'
        return info
    bc = resolve(bsdf.inputs['Base Color'] if 'Base Color' in bsdf.inputs else bsdf.inputs['Color'])
    src = linked_from(bc)
    if src is None:
        info['color'] = as_colour(bc.default_value)
    elif src.type == 'TEX_IMAGE' and src.image:
        info['albedo'] = ('image', src.image)
        info['uv_scale'] = mapping_scale(src)
        info['color'] = [1, 1, 1, 1]
    elif src.type == 'TEX_BRICK':
        info['albedo'] = ('brick', src)
        info['color'] = [1, 1, 1, 1]
        info['note'] = 'Brick Texture made into a seamless tile'
    else:
        # something in between (Mix, Hue/Saturation, Gamma, a group...): use an image
        # texture further up if there is one, else a colour it holds
        img_node = upstream(src, lambda n: n.type == 'TEX_IMAGE' and n.image)
        if img_node:
            info['albedo'] = ('image', img_node.image)
            info['uv_scale'] = mapping_scale(img_node)
            info['color'] = [1, 1, 1, 1]
            info['note'] = 'image texture behind a %s node (that node is left out)' % src.type
        else:
            info['color'] = guess_colour(src)
            info['note'] = '%s base colour: flat colour' % src.type
    for k in ('Roughness', 'Metallic'):
        if k in bsdf.inputs and not resolve(bsdf.inputs[k]).is_linked:
            info[k.lower()] = float(sock_value(bsdf.inputs[k]))
        elif k == 'Roughness':
            info['roughness'] = 0.5
    a = bsdf.inputs.get('Alpha')
    a = resolve(a) if a is not None else None
    if a is not None and (a.is_linked or a.default_value < 0.99):
        info['alpha'] = True
    return info


game_mats = {}      # source material (or None) -> dict(name, mat, info)


def game_material(src):
    key = src.name if src else None
    if key in game_mats:
        return game_mats[key]
    info = analyse_material(src)
    gname = 'zone_' + slug(src.name if src else 'default')
    while any(k.lower() in gname.lower() for k in GAME_LIBRARY_NAMES) or \
            any(g['name'] == gname for g in game_mats.values()):
        gname += '_m'
    m = bpy.data.materials.new(gname)
    m.use_nodes = True
    nt = m.node_tree
    p = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    # what the bake sees: the colour, matte (only diffuse light is baked)
    p.inputs['Metallic'].default_value = 0.0
    p.inputs['Roughness'].default_value = 1.0
    tex_path = None
    if info['albedo'] and info['albedo'][0] == 'image' and min(info['albedo'][1].size) == 0:
        log('%s: its image %s could not be read (missing file?) - light grey instead' % (
            src.name if src else '?', info['albedo'][1].filepath))
        info['albedo'] = None
        info['color'] = [0.8, 0.8, 0.8, 1.0]
        info['note'] = 'image missing: light grey'
    if info['albedo']:
        kind, obj = info['albedo']
        if kind == 'brick':
            rgb, uv_scale, n = brick_tile(obj)
            img, tex_path = save_png(gname + '_albedo', rgb)
            info['uv_scale'] = (info['uv_scale'][0] * uv_scale[0], info['uv_scale'][1] * uv_scale[1])
        else:
            img = obj
            tex_path = os.path.join(OUT, 'textures', gname + '_albedo.png')
            w, h = img.size
            px = np.empty(w * h * 4, np.float32)
            img.pixels.foreach_get(px)
            px = np.flipud(px.reshape(h, w, 4))
            lin = px[..., :3] if img.colorspace_settings.name != 'sRGB' else \
                np.where(px[..., :3] <= 0.04045, px[..., :3] / 12.92, ((px[..., :3] + 0.055) / 1.055) ** 2.4)
            img, tex_path = save_png(gname + '_albedo', lin, px[..., 3] if info['alpha'] else None)
        t = nt.nodes.new('ShaderNodeTexImage')
        t.image = img
        nt.links.new(t.outputs['Color'], p.inputs['Base Color'])
    else:
        c = info['color']
        p.inputs['Base Color'].default_value = (max(c[0], 0.02), max(c[1], 0.02), max(c[2], 0.02), 1)
    if info['alpha']:
        p.inputs['Base Color'].default_value = (1, 1, 1, 1)   # its own light is baked white
    if info['emission']:                                        # lamps keep lighting the bake
        p.inputs['Emission Color'].default_value = info['emission'][0]
        p.inputs['Emission Strength'].default_value = info['emission'][1]
    target = nt.nodes.new('ShaderNodeTexImage')
    target.name = 'ZoneBakeTarget'
    g = {'name': gname, 'mat': m, 'info': info, 'texture': tex_path, 'source': key}
    game_mats[key] = g
    return g


# ---------------------------------------------------------------- geometry
def evaluated_mesh(o, decimate=None, disable=()):
    saved = []
    for mod in disable:
        saved.append((mod, mod.show_viewport, mod.show_render))
        mod.show_viewport = mod.show_render = False
    dec = None
    if decimate:
        dec = o.modifiers.new('ZoneDecimate', 'DECIMATE')
        dec.ratio = decimate
        dec.use_collapse_triangulate = True
        o.modifiers.move(len(o.modifiers) - 1, 0)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    if dec:
        o.modifiers.remove(dec)
    for mod, v, r in saved:
        mod.show_viewport, mod.show_render = v, r
    return me


def set_materials(me, mats):
    """swap a mesh's materials slot by slot (materials.clear() would send every face to slot 0)"""
    for i, m in enumerate(mats):
        if i < len(me.materials):
            me.materials[i] = m
        else:
            me.materials.append(m)


def tris(me):
    return sum(len(p.vertices) - 2 for p in me.polygons)


def area(me):
    return sum(p.area for p in me.polygons)


def pick_size(a):
    need = math.sqrt(max(a, 1e-6) * DENSITY * DENSITY / PACKING)
    sizes = [s for s in LIGHTMAP_SIZES if MIN_SIZE <= s <= MAX_SIZE]
    return min(sizes, key=lambda s: abs(math.log(s / need)))


def prepare_uvs(me, gm):
    """UV map 0: the one the materials use, scaled to the game textures; nothing else"""
    names = [u.name for u in me.uv_layers]
    keep = next((u.name for u in me.uv_layers if u.active_render), names[0] if names else None)
    for n in names:
        if n != keep:
            me.uv_layers.remove(me.uv_layers[n])      # (look layers up again after each change)
    if keep is None:
        me.uv_layers.new(name='UVMap')
    else:
        me.uv_layers[0].name = 'UVMap'
    lay = me.uv_layers[0]
    uv = np.empty(len(me.loops) * 2, np.float32)
    lay.data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2)
    for poly in me.polygons:
        s = gm[min(poly.material_index, len(gm) - 1)]['info']['uv_scale']
        if s != (1.0, 1.0):
            for li in poly.loop_indices:
                uv[li, 0] *= s[0]
                uv[li, 1] *= s[1]
    me.uv_layers[0].data.foreach_set('uv', uv.ravel())


def lightmap_uv(me, size):
    """UV map 1, 'zone_lightmap': every face its own place, 4 texels apart"""
    me.uv_layers.new(name='zone_lightmap')
    me.uv_layers.active_index = 1
    tmp = bpy.data.objects.new('zone_unwrap', me)
    coll.objects.link(tmp)
    for o2 in view_layer.objects:
        o2.select_set(False)
    view_layer.objects.active = tmp
    tmp.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.0, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.uv.pack_islands(rotate=True, margin_method='FRACTION', margin=4.0 / size)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects.remove(tmp)
    me.uv_layers.active_index = 0
    me.uv_layers[0].active_render = True


def two_sided(me, label):
    """Blender draws both sides of a face; the game draws only the front, and Cycles bakes
    only the front. A closed mesh (a box, a column) gets its faces pointing out; an open
    one (a single-layer wall, floor or water sheet) gets a second, turned-round copy of
    every face 2 mm behind it, so each side is drawn and lit on its own."""
    if len(me.polygons) > TWO_SIDED_MAX_FACES:
        return 'left as it is'
    bm = bmesh.new()
    bm.from_mesh(me)
    closed = all(e.is_manifold for e in bm.edges) and not any(e.is_boundary for e in bm.edges)
    if closed:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        what = 'closed: faces pointing out'
    else:
        orig = bm.faces[:]
        dup = bmesh.ops.duplicate(bm, geom=orig)
        faces = [g for g in dup['geom'] if isinstance(g, bmesh.types.BMFace)]
        verts = [g for g in dup['geom'] if isinstance(g, bmesh.types.BMVert)]
        bm.normal_update()
        # each copied vertex 2 mm behind the faces it belongs to
        for v in verts:
            n = mathutils.Vector((0, 0, 0))
            for f in v.link_faces:
                n += f.normal
            if n.length > 1e-9:
                v.co -= n.normalized() * 0.002
        bmesh.ops.reverse_faces(bm, faces=faces)
        what = 'open: made two-sided (%d faces)' % (len(orig) * 2)
    bm.to_mesh(me)
    bm.free()
    log('%s: %s' % (label, what))
    return what


coll = bpy.data.collections.new('Zone Map')
scene.collection.children.link(coll)
sources = [o for o in scene.objects if o.type == 'MESH' and renders(o)]
objects = []     # what bake_info.json lists

# instances (geometry nodes, collection instances, particles): what renders that no object owns
def merge_meshes(parts, name):
    """one mesh from (mesh, world matrix) pairs; every mesh has UV map 0 and game materials"""
    mats, midx = [], {}
    cos, lvs, lss, uvs, mis, sms = [], [], [], [], [], []
    vbase = lbase = 0
    for me, M in parts:
        nv, nl, npl = len(me.vertices), len(me.loops), len(me.polygons)
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3) @ np.array(M.to_3x3(), np.float32).T + np.array(M.translation, np.float32)
        lv = np.empty(nl, np.int32)
        me.loops.foreach_get('vertex_index', lv)
        ls = np.empty(npl, np.int32)
        me.polygons.foreach_get('loop_start', ls)
        lt = np.empty(npl, np.int32)
        me.polygons.foreach_get('loop_total', lt)
        uv = np.zeros(nl * 2, np.float32)
        lay = next((u for u in me.uv_layers if u.active_render), None) or (me.uv_layers[0] if len(me.uv_layers) else None)
        if lay is not None:
            lay.data.foreach_get('uv', uv)
        uv = uv.reshape(-1, 2)
        mi = np.empty(npl, np.int32)
        me.polygons.foreach_get('material_index', mi)
        sm = np.empty(npl, bool)
        me.polygons.foreach_get('use_smooth', sm)
        remap = []
        for m in list(me.materials) or [None]:
            m = m.original if m is not None else None      # (an evaluated mesh has evaluated copies)
            k = m.name if m else ''
            if k not in midx:
                midx[k] = len(mats)
                mats.append(m)
            remap.append(midx[k])
        mi = np.array(remap, np.int32)[np.minimum(mi, len(remap) - 1)]
        if M.determinant() < 0:            # mirrored: turn each face's loops round
            p = np.repeat(np.arange(npl), lt)
            rev = 2 * ls[p] + lt[p] - 1 - np.arange(nl)
            lv, uv = lv[rev], uv[rev]
        cos.append(co); lvs.append(lv + vbase); lss.append(ls + lbase); uvs.append(uv); mis.append(mi); sms.append(sm)
        vbase += nv
        lbase += nl
    out = bpy.data.meshes.new(name)
    out.vertices.add(vbase)
    out.vertices.foreach_set('co', np.concatenate(cos).ravel())
    out.loops.add(lbase)
    out.loops.foreach_set('vertex_index', np.concatenate(lvs))
    ls = np.concatenate(lss)
    out.polygons.add(len(ls))
    out.polygons.foreach_set('loop_start', ls)
    out.uv_layers.new(name='UVMap').data.foreach_set('uv', np.concatenate(uvs).ravel())
    out.polygons.foreach_set('material_index', np.concatenate(mis))
    out.polygons.foreach_set('use_smooth', np.concatenate(sms))
    for m in mats:                      # before validate(), which resets material indices
        out.materials.append(m)         # past the number of slots
    out.update(calc_edges=True)
    out.validate()
    return out


uniq, placed = {}, []
_dg = bpy.context.evaluated_depsgraph_get()
for inst in _dg.object_instances:
    if not inst.is_instance or inst.object.type != 'MESH':
        continue
    par = inst.parent.original if inst.parent else None
    if par is not None and not renders(par):
        continue
    key = inst.object.data.as_pointer()
    if key not in uniq:
        if not len(inst.object.data.polygons):
            uniq[key] = None
            continue
        # copy the instanced geometry itself (new_from_object would rebuild the instancer)
        me = merge_meshes([(inst.object.data, mathutils.Matrix.Identity(4))], 'zone_instance')
        mats = list(me.materials) or [None]
        uniq[key] = {'mesh': me, 'mats': mats, 'tris': tris(me), 'count': 0, 'name': inst.object.data.name}
    if uniq[key] is None:
        continue
    uniq[key]['count'] += 1
    placed.append((key, inst.matrix_world.copy(), par.name if par else 'instances'))
uniq = {k: v for k, v in uniq.items() if v}
if placed:
    log('%d instances of %d shapes (%d triangles in all)' % (
        len(placed), len(uniq), sum(u['tris'] * u['count'] for u in uniq.values())))


def split_by_alpha(me, gm):
    """[(name tag, mesh, its materials)]: see-through faces (glass, lenses, water) apart from
    the rest, so only they are left out of the light's way and out of the collision"""
    alpha = [g['info']['alpha'] for g in gm]
    mi = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get('material_index', mi)
    used = np.array(alpha)[np.minimum(mi, len(alpha) - 1)] if len(mi) else np.zeros(0, bool)
    if not used.any() or used.all():               # (by the faces, not the material slots)
        return [('', me, gm if used.all() else [g for g in gm if not g['info']['alpha']] or gm)]
    clear = me.copy()
    for mesh, want in ((me, False), (clear, True)):
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if alpha[min(f.material_index, len(alpha) - 1)] != want],
                         context='FACES')
        bm.to_mesh(mesh)
        bm.free()
    return [('', me, [g for g in gm if not g['info']['alpha']]),
            ('_clear', clear, [g for g in gm if g['info']['alpha']])]


def add_piece(me, name, source, gm, size, a_m2):
    """one lightmapped object in the map"""
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    see_through = any(g['info']['alpha'] for g in gm)
    objects.append({'name': ob.name, 'source': source, 'size': size, 'area_m2': round(a_m2, 2),
                    'texel_cm': round(100 * math.sqrt(a_m2 / (size * size * PACKING)), 2),
                    'see_through': see_through, 'materials': [g['name'] for g in gm]})
    if see_through:
        # not in the way of the light
        for k in ('visible_diffuse', 'visible_glossy', 'visible_transmission', 'visible_shadow',
                  'visible_volume_scatter'):
            setattr(ob, k, False)
        if not SEE_THROUGH_SOLID:
            # nothing to collide with in the game: its collision moved 1 km down
            cme = bpy.data.meshes.new(name + '-collider')
            cme.from_pydata([(0, 0, -1000), (0.01, 0, -1000), (0, 0.01, -1000)], [], [(0, 1, 2)])
            cme.materials.append(gm[0]['mat'])
            coll.objects.link(bpy.data.objects.new(name + '-collider', cme))
    return ob
for o in sources:
    arrays = [m for m in o.modifiers if m.type == 'ARRAY' and m.show_render]
    usable = len(arrays) == 1 and arrays[0].fit_type == 'FIXED_COUNT' and not arrays[0].use_object_offset
    base = evaluated_mesh(o, disable=arrays if usable else ())
    if len(base.polygons) == 0:
        bpy.data.meshes.remove(base)
        continue
    copies, offset = 1, mathutils.Vector((0, 0, 0))
    if usable:
        a = arrays[0]
        lo = mathutils.Vector([min(v.co[i] for v in base.vertices) for i in range(3)])
        hi = mathutils.Vector([max(v.co[i] for v in base.vertices) for i in range(3)])
        if a.use_relative_offset:
            offset += mathutils.Vector([(hi - lo)[i] * a.relative_offset_displace[i] for i in range(3)])
        if a.use_constant_offset:
            offset += mathutils.Vector(a.constant_offset_displace)
        copies = a.count
    elif arrays:
        log('%s: its array is not a fixed count with relative/constant offsets, kept whole' % o.name)
    t = tris(base)
    if t > COLUMN_TRIS * 2:
        log('%s: %d triangles a copy, decimating to %d' % (o.name, t, COLUMN_TRIS))
        bpy.data.meshes.remove(base)
        base = evaluated_mesh(o, decimate=COLUMN_TRIS / t, disable=arrays if usable else ())
    src_mats = [s.material for s in o.material_slots] or [None]
    gm = [game_material(m) for m in src_mats]
    base.transform(o.matrix_world)            # world space, as copy 0
    if o.matrix_world.determinant() < 0:
        base.flip_normals()
    set_materials(base, [g['mat'] for g in gm])
    two_sided(base, o.name)
    prepare_uvs(base, gm)
    step = o.matrix_world.to_3x3() @ offset
    for tag, pme, pgm in split_by_alpha(base, gm):
        a_m2 = area(pme)
        size = pick_size(a_m2)
        lightmap_uv(pme, size)        # once: the copies are the same shape
        for i in range(copies):
            me = pme.copy() if i < copies - 1 else pme
            if i:
                me.transform(mathutils.Matrix.Translation(step * i))
            add_piece(me, '%s%s_%02d_zlms-%d' % (slug(o.name), tag, i, size), o.name, pgm, size, a_m2)
    o.hide_render = True
    o.hide_set(True)
    log('%s: %d piece(s), lightmaps %dx%d (%.1f cm a texel)' % (o.name, copies, size, size,
                                                                 objects[-1]['texel_cm']))


def hull_mesh(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    res = bmesh.ops.convex_hull(bm, input=bm.verts[:])
    bmesh.ops.delete(bm, geom=list(set(res['geom_interior']) | set(res['geom_unused'])), context='VERTS')
    for f in bm.faces:
        f.material_index = 0
    out = bpy.data.meshes.new(me.name + '_hull')
    bm.to_mesh(out)
    bm.free()
    out.uv_layers.new(name='UVMap')
    for m in me.materials:
        out.materials.append(m)
    return out


def decimate_mesh(me, ratio):
    tmp = bpy.data.objects.new('zone_decimate', me)
    coll.objects.link(tmp)
    mod = tmp.modifiers.new('ZoneDecimate', 'DECIMATE')
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    new = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    bpy.data.objects.remove(tmp)
    return new


if placed:
    total = sum(u['tris'] * u['count'] for u in uniq.values())
    f = min(1.0, INSTANCE_TRIS / max(total, 1))
    share = INSTANCE_TRIS / len(uniq)          # each shape the same share, whatever its count:
    for u in uniq.values():                    # a few lamps keep their detail, 3000 grid bars don't
        target = max(12, int(share / u['count']))
        if f < 1.0 and u['tris'] > target:
            new = decimate_mesh(u['mesh'], target / u['tris'])
            if tris(new) > 3 * target:
                # many little separate parts that will not collapse: its outer shape instead
                bpy.data.meshes.remove(new)
                hull = hull_mesh(u['mesh'])
                new = decimate_mesh(hull, min(1.0, 2 * target / max(tris(hull), 1)))
                bpy.data.meshes.remove(hull)
                log('%s: decimating stalled, using its convex hull (%d triangles) x%d' % (
                    u['name'], tris(new), u['count']))
            bpy.data.meshes.remove(u['mesh'])
            u['mesh'], u['tris'] = new, tris(new)
        gm = [game_material(m) for m in u['mats']]
        set_materials(u['mesh'], [g['mat'] for g in gm])
        prepare_uvs(u['mesh'], gm)
        u['gm'] = gm
    if f < 1.0:
        log('instances decimated to %d triangles in all (INSTANCE_TRIS)' % sum(u['tris'] * u['count'] for u in uniq.values()))
    cells = {}
    for key, M, par in placed:
        t = M.translation
        cell = (par, int(math.floor(t.x / INSTANCE_CELL)), int(math.floor(t.y / INSTANCE_CELL)),
                int(math.floor(t.z / INSTANCE_CELL)))
        cells.setdefault(cell, []).append((key, M))
    for n_cell, (cell, items) in enumerate(sorted(cells.items())):
        me = merge_meshes([(uniq[k]['mesh'], M) for k, M in items], 'zone_cell')
        gm = [next(g for u in uniq.values() for g in u['gm'] if g['mat'] == m) for m in me.materials]
        for tag, pme, pgm in split_by_alpha(me, gm):
            a_m2 = area(pme)
            size = pick_size(a_m2)
            lightmap_uv(pme, size)
            add_piece(pme, '%s%s_%03d_zlms-%d' % (slug(cell[0]), tag, n_cell, size), cell[0], pgm, size, a_m2)
    log('instances: %d pieces of up to %g m' % (len(cells), INSTANCE_CELL))
    for u in uniq.values():
        bpy.data.meshes.remove(u['mesh'])
    for o in scene.objects:            # the instancers are done with
        if o.name not in coll.objects and any(m.type == 'NODES' for m in getattr(o, 'modifiers', [])) \
                or o.instance_type != 'NONE':
            o.hide_render = True
            o.hide_set(True)

pieces = [(bpy.data.objects[x['name']], x['source']) for x in objects]
if os.environ.get('ZONE_BAKE_DRY'):
    bpy.ops.wm.save_mainfile()
    raise SystemExit('dry run: %d pieces' % len(objects))


# ---------------------------------------------------------------- portals
portals = 0
portal_at = []
if PORTALS:
    dg = bpy.context.evaluated_depsgraph_get()
    for ob, _ in pieces:
        if not ob.visible_shadow:
            continue
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        seen = set()
        for e in bm.edges:
            if not e.is_boundary or e.index in seen:
                continue
            loop, stack = [], [e]
            while stack:
                x = stack.pop()
                if x.index in seen:
                    continue
                seen.add(x.index)
                loop.append(x)
                for v in x.verts:
                    stack += [y for y in v.link_edges if y.is_boundary and y.index not in seen]
            pts = [v.co for x in loop for v in x.verts]
            lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)])
            hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
            d = hi - lo
            if d.z > 0.05 * max(d.x, d.y) or d.x < 0.3 or d.y < 0.3:
                continue        # not a horizontal opening
            c = (lo + hi) / 2
            if any((c - p).length < 0.05 for p in portal_at):
                continue        # the other side of a two-sided opening
            hit = scene.ray_cast(dg, c + mathutils.Vector((0, 0, 0.05)), (0, 0, 1))[0]
            if hit:
                continue        # something above it: not open to the sky
            ld = bpy.data.lights.new('zone_portal', 'AREA')
            ld.shape = 'RECTANGLE'
            ld.size, ld.size_y = d.x, d.y
            ld.cycles.is_portal = True
            lo_ = bpy.data.objects.new('zone_portal', ld)
            lo_.location = c       # an area light faces -Z: into the room
            coll.objects.link(lo_)
            portal_at.append(c)
            portals += 1
        bm.free()
log('%d light portals in openings to the sky' % portals)


# ---------------------------------------------------------------- bake
device = setup_device()
log('baking on', device)
cy = scene.cycles
cy.samples = SAMPLES
cy.use_adaptive_sampling = False
cy.use_denoising = False
cy.diffuse_bounces = DIFFUSE_BOUNCES
cy.max_bounces = max(cy.max_bounces, DIFFUSE_BOUNCES)
cy.sample_clamp_indirect = 10.0
st = scene.render.image_settings          # lightmaps are saved as 16-bit float EXR (linear)
st.file_format = 'OPEN_EXR'
st.color_depth = '16'
st.exr_codec = 'ZIP'
st.color_mode = 'RGB'
bk = scene.render.bake
bk.margin = 4
bk.margin_type = 'EXTEND'


def bake_one(info):
    """one object's light, into its own float image, saved as EXR"""
    ob = bpy.data.objects[info['name']]
    img = bpy.data.images.new(ob.name, info['size'], info['size'], alpha=False, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    for m in ob.data.materials:
        n = m.node_tree.nodes['ZoneBakeTarget']
        n.image = img
        m.node_tree.nodes.active = n
    for o2 in view_layer.objects:
        o2.select_set(False)
    ob.select_set(True)
    view_layer.objects.active = ob
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'DIRECT', 'INDIRECT'}, uv_layer='zone_lightmap',
                        margin=4, margin_type='EXTEND', use_clear=True, target='IMAGE_TEXTURES')
    path = os.path.join(OUT, 'lightmaps', ob.name + '.exr')
    if not (DENOISE and denoise(img, path)):
        img.save_render(path, scene=scene)     # half-float EXR
    bpy.data.images.remove(img)


_dn = {}


def denoise(img, path):
    """Blender's denoiser on a baked image, through the compositor (no scene render: the
    compositor has no Render Layers). Writes the EXR; False if it could not."""
    global DENOISE
    try:
        if not _dn:
            tree = bpy.data.node_groups.new('zone_denoise', 'CompositorNodeTree')
            tree.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
            i = tree.nodes.new('CompositorNodeImage')
            d = tree.nodes.new('CompositorNodeDenoise')
            if 'HDR' in d.inputs:
                d.inputs['HDR'].default_value = True
            o = tree.nodes.new('NodeGroupOutput')
            tree.links.new(i.outputs['Image'], d.inputs['Image'])
            tree.links.new(d.outputs['Image'], o.inputs[0])
            _dn.update(tree=tree, image=i)
        _dn['image'].image = img
        scene.compositing_node_group = _dn['tree']
        scene.render.use_compositing = True
        scene.render.resolution_x, scene.render.resolution_y = img.size
        scene.render.resolution_percentage = 100
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(path, scene=scene)
        out = bpy.data.images.load(path, check_existing=False)
        px = np.empty(out.size[0] * out.size[1] * 4, np.float32)
        out.pixels.foreach_get(px)
        if (px < 0).any() or not np.isfinite(px).all():    # the game takes no negative light
            out.pixels.foreach_set(np.nan_to_num(np.maximum(px, 0)))
            out.save_render(path, scene=scene)
        bpy.data.images.remove(out)
        return True
    except Exception as e:
        log('denoising off - Blender\'s compositor did not take it: %r' % (e,))
        DENOISE = False
        return False


total_texels = sum(o['size'] ** 2 for o in objects)
done = 0
t_bake = time.time()
log('%d lightmaps, %.1f M texels' % (len(objects), total_texels / 1e6))
for i, info in enumerate(objects):
    t0 = time.time()
    bake_one(info)
    done += info['size'] ** 2
    left = (time.time() - t_bake) / done * (total_texels - done)
    log('%d/%d %s (%dx%d) in %.0f s - about %d min left' % (
        i + 1, len(objects), info['name'], info['size'], info['size'], time.time() - t0, left / 60 + 0.5))

# ---------------------------------------------------------------- export
for g in game_mats.values():
    n = g['mat'].node_tree.nodes.get('ZoneBakeTarget')
    if n:
        g['mat'].node_tree.nodes.remove(n)
for o2 in view_layer.objects:
    o2.select_set(False)
for o2 in coll.objects:
    if o2.type == 'MESH':
        o2.select_set(True)
glb = os.path.join(OUT, 'map.glb')
bpy.ops.export_scene.gltf(filepath=glb, export_format='GLB', use_selection=True, export_apply=False,
                          export_texcoords=True, export_normals=True, export_tangents=False,
                          export_materials='EXPORT', export_image_format='NONE', export_yup=True,
                          export_cameras=False, export_lights=False, export_extras=False)

# occluder.bin: the simple opaque pieces (walls, floors), for the game's occlusion culling
import struct
occ_v, occ_i = [], []
for x in objects:
    ob = bpy.data.objects[x['name']]
    me = ob.data
    if x['see_through'] or tris(me) > 2000:
        continue
    me.calc_loop_triangles()
    base_i = len(occ_v)
    occ_v += [(v.co.x, v.co.z, -v.co.y) for v in me.vertices]
    for t in me.loop_triangles:
        occ_i += [base_i + t.vertices[0], base_i + t.vertices[1], base_i + t.vertices[2]]
with open(os.path.join(OUT, 'occluder.bin'), 'wb') as fp:
    fp.write(b'ZOCC' + struct.pack('<III', 1, len(occ_v), len(occ_i)))
    fp.write(np.array(occ_v, '<f4').tobytes())
    fp.write(np.array(occ_i, '<i4').tobytes())
log('occluder: %d triangles' % (len(occ_i) // 3))

# spawn: an empty with "spawn" in its name (facing along its green Y arrow), else the scene
# camera (Blender Z-up -> the game's Y-up)
spawn = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
mark = next((o for o in scene.objects if o.type == 'EMPTY' and 'spawn' in o.name.lower()), None)
cam = mark or scene.camera
if cam:
    p = cam.matrix_world.translation
    f = cam.matrix_world.to_3x3() @ mathutils.Vector((0, 1, 0) if mark else (0, 0, -1))
    fg = (f.x, f.z, -f.y)
    spawn = [p.x, p.z, -p.y, 0.0, math.atan2(-fg[0], -fg[2]), 0.0, 1.0, 1.0, 1.0]

info = {
    'version': 1, 'map': NAME, 'blend': os.path.basename(blend), 'blender': bpy.app.version_string,
    'density': DENSITY, 'samples': SAMPLES, 'device': device, 'portals': portals,
    'variation': VARIATION,
    'lighting_variation': {'skybox': {'skybox_name': SKYBOX, 'sky_rotation': SKY_ROTATION,
                                      'energy': SKY_ENERGY}},
    'spawn_point': spawn,
    'objects': objects,
    'materials': {g['name']: {'color': [round(float(c), 4) for c in g['info']['color']],
                              'roughness': round(g['info']['roughness'], 3),
                              'metallic': round(g['info']['metallic'], 3),
                              'alpha': g['info']['alpha'],
                              'emission_color': [round(float(c), 4) for c in g['info']['emission'][0]] if g['info']['emission'] else None,
                              'emission_energy': round(g['info']['emission'][1], 4) if g['info']['emission'] else 0.0,
                              'albedo': os.path.relpath(g['texture'], OUT).replace('\\', '/') if g['texture'] else None,
                              'source': g['source'], 'note': g['info']['note']}
                  for g in game_mats.values()},
    'minutes': round((time.time() - t_start) / 60, 1),
}
with open(os.path.join(OUT, 'bake_info.json'), 'w') as fp:
    json.dump(info, fp, indent=1)
bpy.ops.wm.save_mainfile()
log('baked in %.1f min: %s' % ((time.time() - t_start) / 60, OUT))


# ---------------------------------------------------------------- pack
def find_game():
    for d in ([GAME_DIR] if GAME_DIR else []) + [
            r'C:\Program Files (x86)\Steam\steamapps\common\The Zone FPV',
            r'C:\Program Files\Steam\steamapps\common\The Zone FPV']:
        if d and os.path.isfile(os.path.join(d, 'thezone.pck')):
            return d
    return GAME_DIR or ''


def find_godot():
    if GODOT_EXE:
        return GODOT_EXE
    import glob
    home = os.path.expanduser('~')
    found = []
    for root in (os.path.join(home, 'Downloads'), os.path.join(home, 'Desktop'), r'C:\Program Files\Godot'):
        for depth in ('', '*', '*/*'):
            found += glob.glob(os.path.join(root, depth, 'Godot_v4.*_win64*.exe'))
    found = [f for f in found if os.path.isfile(f) and not re.search(r'v4\.[0-3][.-]', f)]
    found.sort(key=lambda f: ('console' not in f, f))   # the console one prints progress
    return found[0] if found else ''


with open(os.path.join(OUT, 'pack_zone_map.gd'), 'w', newline='\n') as fp:
    fp.write(PACKER_GD)
game, godot = find_game(), find_godot()
map_dir = os.path.join(game, 'custom_maps', NAME) if game else ''
with open(os.path.join(OUT, 'pack.bat'), 'w', newline='\r\n') as fp:
    fp.write('@echo off\n'
             'rem Packs this folder into The Zone FPV\'s custom_maps (made by zone_bake.py).\n'
             'set "GODOT=%s"\n'
             'set "MAP=%s"\n'
             'if not exist "%%GODOT%%" (echo Set GODOT in this file to a Godot 4.4+ editor .exe & pause & exit /b 1)\n'
             'if "%%MAP%%"=="" (echo Set MAP in this file to ...\\The Zone FPV\\custom_maps\\%s & pause & exit /b 1)\n'
             '"%%GODOT%%" --headless --script "%%~dp0pack_zone_map.gd" -- "%%~dp0." "%%MAP%%"\n'
             'pause\n' % (godot, map_dir, NAME))
if not AUTO_PACK:
    log('to pack: double-click', os.path.join(OUT, 'pack.bat'))
elif not godot or not os.path.isfile(godot):
    log('no Godot 4.4+ found: download the editor from godotengine.org, set GODOT_EXE (or GODOT in '
        'zone_bake/pack.bat) and double-click zone_bake/pack.bat')
elif not game:
    log('no game folder found: set GAME_DIR (or MAP in zone_bake/pack.bat) and double-click zone_bake/pack.bat')
else:
    import subprocess
    log('packing with', godot)
    p = subprocess.Popen([godot, '--headless', '--script', os.path.join(OUT, 'pack_zone_map.gd'), '--',
                          OUT, map_dir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(p.stdout.readline, b''):          # bytes: Windows code pages can't decode Godot's
        line = line.decode('utf-8', 'replace').rstrip()
        if line and not line.startswith('Godot Engine'):
            log('  ' + line)
    if p.wait() == 0:
        log('installed: %s - untick Legacy lighting in the beta and pick %s from the custom maps' % (map_dir, NAME))
    else:
        log('packing failed (above); fix it and double-click zone_bake/pack.bat')
log('all done in %.1f min' % ((time.time() - t_start) / 60))
