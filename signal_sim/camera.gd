extends Node
## signal_sim camera: the bloom of a mode's lens (bloom, bloom_size), while that
## mode is on (Analog 400mw by Nito's whoop camera).
##
## The map's glow is turned on and set up like a cheap whoop camera's
## lens - light scatters from everything bright into what is around it, so
## sky, sunlit walls, windows and lamps spill past their edges with a soft
## halo. It is rendered in the 3D view, so it comes from what is really
## bright and never from the OSD. The haze and purple fringe that go with it
## are in the mode's shader. (The 3D view itself renders at the game's own
## resolution: the mode's shaders work out the analog picture's detail.)
##
## The game's own render and glow settings come back in the other video modes.
## main.gd hands over the mode's values with set_mode() ({} = no mode of ours).

const DEFAULTS := {"bloom": 0.0, "bloom_size": 1.0}

# The glow is a lens's scatter: every pixel gives up a share of its light
# (GLARE_MIX at glare=1) to a blur of itself, so a halo appears only where
# bright meets darker and a big even area - a white plaza, a sunlit roof, the
# sky - stays as it was instead of flooding the picture. How far it reaches
# is set by the glow levels (Environment glow levels 1-7, for a picture 480
# render pixels tall): a tight overflow at levels 2-3 and a halo out to
# level 6. A spot counts at most GLOW_CAP times white, which keeps a
# sun-baked surface from swamping everything near it.
const GLARE_MIX := 0.3
const GLOW_CAP := 4.0
const GLOW_LEVELS := [0.0, 1.0, 1.0, 0.7, 0.5, 0.3, 0.0]
const GLOW_PROPS := ["glow_enabled", "glow_intensity", "glow_strength", "glow_bloom", "glow_blend_mode",
	"glow_hdr_threshold", "glow_hdr_scale", "glow_hdr_luminance_cap", "glow_normalized", "glow_mix", "glow_map_strength"]

var cfg := {}
var applied_scale := -1.0      # for tests
var applied_glow := false      # for tests

var _scene: Node = null
var _sv: SubViewport = null
var _pp: CanvasItem = null
var _orig := {}
var _t_retry := 0.0
var _glow_on := true
var _env: Environment = null
var _env_orig := {}


## The active mode's camera values, or {} when none of signal_sim's modes is
## on. Returns a line for status.txt.
func set_mode(values: Dictionary) -> String:
	if values.is_empty():
		if not cfg.is_empty():
			_restore()
		cfg = {}
		return "camera: off"
	var want := DEFAULTS.duplicate()
	for k in values:
		if want.has(k):
			want[k] = values[k]
	if want != cfg:
		_restore()
		cfg = want
	_glow_on = float(cfg["bloom"]) > 0.0
	var parts := PackedStringArray()
	# the glow this is built on washes the picture out white on the Compatibility
	# renderer (OpenGL, which Godot falls back to without Vulkan)
	if _glow_on and RenderingServer.get_current_rendering_method() == "gl_compatibility":
		_glow_on = false
		parts.append("no bloom (the game is on OpenGL, where it washes the picture out)")
	else:
		parts.append("bloom %s" % str(cfg["bloom"]) if _glow_on else "no bloom")
	return "camera: " + ", ".join(parts)


func _process(delta: float) -> void:
	var cs := get_tree().current_scene
	_t_retry += delta
	if cs != _scene or (_sv == null and _t_retry > 1.0):
		_restore()
		_scene = cs
		_t_retry = 0.0
		_find()
	if _sv == null or not is_instance_valid(_sv) or not is_instance_valid(_pp):
		_sv = null
		_env = null
		return
	var m = _pp.material
	var crop := 0.25
	if m is ShaderMaterial:
		var c = m.get_shader_parameter("fisheye_crop")
		if c != null:
			crop = float(c)
	if cfg.is_empty() or not _glow_on:
		_restore()
		# not overriding: follow whatever the game has set
		_orig = {"scale": _sv.scaling_3d_scale, "mode": _sv.scaling_3d_mode, "msaa": _sv.msaa_3d,
			"bias": _sv.texture_mipmap_bias}
		return
	var h := get_viewport().get_visible_rect().size.y
	if h < 100.0:
		return
	if _glow_on:
		_glow(crop)


## The environment the 3D view renders with: the camera's own, else the
## map's WorldEnvironment, else the world's fallback.
func _find_env() -> Environment:
	var cam := _sv.get_camera_3d()
	if cam != null and cam.environment != null:
		return cam.environment
	var w := _sv.find_world_3d()
	if w == null:
		return null
	return w.environment if w.environment != null else w.fallback_environment


func _glow(crop: float) -> void:
	var e := _find_env()
	if e != _env:
		_restore_glow()
		_env = e
		if e == null:
			return
		for p in GLOW_PROPS:
			_env_orig[p] = e.get(p)
		for i in 7:
			_env_orig["level%d" % i] = e.get_glow_level(i)
	if e == null:
		return
	# The glow levels are sized in render pixels: shift them so the halo
	# covers the same part of the picture at any render resolution, then by
	# glare_size.
	var rh := float(_sv.size.y) * _sv.scaling_3d_scale * maxf(1.0 - crop, 0.25)
	var shift := log(maxf(rh, 60.0) / 480.0) / log(2.0) + log(clampf(float(cfg["bloom_size"]), 0.25, 4.0)) / log(2.0)
	for i in 7:
		var x := float(i) - shift
		var i0 := floori(x)
		var f := x - float(i0)
		var w := (1.0 - f) * _lvl(i0) + f * _lvl(i0 + 1)
		if absf(e.get_glow_level(i) - w) > 0.001:
			e.set_glow_level(i, w)
	var want := {"glow_enabled": true, "glow_blend_mode": Environment.GLOW_BLEND_MODE_MIX,
		"glow_mix": clampf(GLARE_MIX * float(cfg["bloom"]), 0.0, 1.0), "glow_intensity": 1.0,
		"glow_strength": 1.0, "glow_normalized": true, "glow_bloom": 1.0, "glow_hdr_threshold": 0.0,
		"glow_hdr_luminance_cap": GLOW_CAP, "glow_map_strength": 0.0}
	for k in want:
		var cur = e.get(k)
		if typeof(cur) == TYPE_FLOAT and absf(cur - float(want[k])) < 0.0005:
			continue
		if cur != want[k]:
			e.set(k, want[k])
	applied_glow = true


func _lvl(i: int) -> float:
	return GLOW_LEVELS[i] if i >= 0 and i < GLOW_LEVELS.size() else 0.0


func _restore_glow() -> void:
	if _env != null and is_instance_valid(_env) and not _env_orig.is_empty():
		for p in GLOW_PROPS:
			_env.set(p, _env_orig[p])
		for i in 7:
			_env.set_glow_level(i, _env_orig["level%d" % i])
	_env = null
	_env_orig = {}
	applied_glow = false


func _find() -> void:
	_sv = null
	_pp = null
	_orig = {}
	if _scene == null:
		return
	_sv = _scene.get_node_or_null("%SubViewport") as SubViewport
	_pp = _scene.get_node_or_null("%PostProcessingShader") as CanvasItem
	if _sv == null or _pp == null:
		_sv = null
		return
	_orig = {"scale": _sv.scaling_3d_scale, "mode": _sv.scaling_3d_mode, "msaa": _sv.msaa_3d,
			"bias": _sv.texture_mipmap_bias}


func _restore() -> void:
	if applied_scale >= 0.0 and _sv != null and is_instance_valid(_sv) and not _orig.is_empty():
		_sv.scaling_3d_scale = _orig["scale"]
		_sv.scaling_3d_mode = _orig["mode"]
		_sv.msaa_3d = _orig["msaa"]
		_sv.texture_mipmap_bias = _orig["bias"]
	applied_scale = -1.0
	_restore_glow()


func _exit_tree() -> void:
	_restore()
