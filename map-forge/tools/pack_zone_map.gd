extends SceneTree
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
		var shared := {}       # lightmap file -> [entry, mean]: objects that split one page share its light
		for ob in info.objects:
			var lm: String = ob.get("lightmap", ob.name)
			if shared.has(lm):
				lms[ob.name] = shared[lm][0]
				means[ob.name] = shared[lm][1]
				continue
			var img := Image.load_from_file(bake + "/lightmaps/" + lm + ".exr")
			if img == null:
				fail("can't read lightmaps/%s.exr" % lm)
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
			shared[lm] = [lms[ob.name], means[ob.name]]
			lf.store_buffer(dds)
		lf.close()
		var texs := _reused_textures(info, out, tier.name, tex_ids)
		if not texs.is_empty():
			print("%-6s textures as they were in the map folder (bake_info reuse_textures)" % tier.name)
			files += ["textures_%s.blob" % tier.name]
		var tf: FileAccess = null if not texs.is_empty() else FileAccess.open(out + "/textures_%s.blob" % tier.name, FileAccess.WRITE)
		for path in (tex_ids if tf != null else {}):
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
		if tf != null:
			tf.close()
			files += ["textures_%s.blob" % tier.name]
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
		files += ["lightmaps_%s.blob" % tier.name, "materials_%s.json" % tier.name]
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


## The map folder's own textures, when bake_info.json says the textures have
## not changed since it was packed (tools/relayout_v3.py moves only the light):
## its textures_<tier>.blob is kept and its index reused, if it has every
## texture this bake names -> the index, or {} to pack them again
func _reused_textures(info: Dictionary, out: String, tier: String, tex_ids: Dictionary) -> Dictionary:
	if not info.get("reuse_textures", false):
		return {}
	var mp := out + "/materials_%s.json" % tier
	if not FileAccess.file_exists(mp) or not FileAccess.file_exists(out + "/textures_%s.blob" % tier):
		return {}
	var old = JSON.parse_string(FileAccess.get_file_as_string(mp))
	if not (old is Dictionary) or not (old.get("textures") is Dictionary):
		return {}
	var t: Dictionary = old.textures
	if t.size() != tex_ids.size():
		return {}
	for path in tex_ids:
		if not t.has(tex_ids[path]):
			return {}
	return t


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
