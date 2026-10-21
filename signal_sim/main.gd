extends Node
## signal_sim 1.1
##
## Signal Simulation, opened up:
##   * layers: a pass of its own drawn on top of the game's own modes - it
##     ships with "Analog 400mw by Nito", analog400mw's breakup over the
##     game's analog modes (a folder in signal_sim/layers/ with a layer.cfg and
##     a shader), which can also calm the game's own link noise and its lens
##     tint, and fade the signal in the open
##   * modes of your own in Pause > Game Settings > Signal Simulation, after
##     the game's: each a folder in signal_sim/modes/ with a mode.cfg and a
##     shader (none ship)
##   * for every mode, the game's included: a floor under the signal and how
##     much walls and range cost (settings.cfg [signal])
##
## A layer is on while a mode it is for is picked (over="analog": the
## game's analog modes) and no mode of ours is: its pass goes in after the
## game's link pass, the same size, over the picture and the OSD, with the
## game's signal quality and analog resolution. game_noise gives the game's
## link pass that share of the signal loss; lens_tint sets the game's camera
## tint between neutral (0) and its own (1); both go back when the layer
## goes off.
##
## How, without replacing any of the game's files:
##   menu    when the game's settings page (menus/pause_menu/game_settings) is
##           made, the modes are added to its Signal Simulation list. Picking
##           one stores "signal_sim:<mode>" in ZSettings.SIGNAL_SIMULATION, as
##           the game stores its own; the game treats an unknown value as a
##           mode without a preset of its own and keeps working out the signal.
##   picture ingame_main draws the 3D view through %PostProcessingShader. While
##           a mode of ours is on, that node gets the mode's own ShaderMaterial
##           (the game keeps setting signal_quality on whatever material it
##           has; the lens values are carried over), %LinkShader, the pass
##           the game draws over the OSD, gets the mode's link_shader if it
##           has one (given link_share of the signal loss), and a pass of our
##           own after it draws the mode's overlay_shader. A pass's shader is
##           the mode's own file or one of the game's (res://...), with one of
##           the game's analog presets as its values if the mode names one.
##           Leaving the mode puts the game's materials back.
##   signal  the game reads ZGamestate.analog_tuning["signal"] (its own
##           /analogtune signal command) as the signal quality for every
##           mode. With a floor, an intensity or a mode's open_range set, this
##           puts the game's quality, so changed, there each physics frame;
##           otherwise it leaves it alone. A value you set with /analogtune
##           yourself wins until you reset it.
##
## Uninstall: delete the signal_sim folder (then pick another Signal Simulation).

const VERSION := "1.1"
const DIR := "res://signal_sim/"
const PREFIX := "signal_sim:"
const SETTINGS_SCRIPT := "res://menus/pause_menu/game_settings.gd"
const ANALOG_PRESETS := "res://ingame/analog_presets.gd"
# set by the game on the post-process material: carried over to ours
const GAME_UNIFORMS := ["signal_quality", "fisheye_strength", "fisheye_crop"]
# a mode's passes: drawn over the 3D view before the OSD (%PostProcessingShader),
# over the picture and the OSD (%LinkShader, the game's analog link pass), and
# after that (a pass of signal_sim's own, the breakup)
const PASSES := ["shader", "link_shader", "overlay_shader"]
# a mode's values for camera.gd (the map's glow), not its shaders
const CAMERA_KEYS := ["bloom", "bloom_size"]
const OLD_MOD := "analog400mw"
# a layer's own controls, in settings.cfg's section named after it (the rest
# of its shader's uniforms can be set in its layer.cfg [settings])
const LAYER_KEYS := ["on", "strength", "game_noise", "lens_tint", "open_range"]
const LAYER_DEFAULTS := {"on": true, "strength": 1.0, "game_noise": 1.0, "lens_tint": 1.0, "open_range": 0.0}

var cfg := {"minimum": 0.0, "intensity": 100.0}
var modes := {}                  # id -> {id, name, order, dir, preset, pass: {pass: Shader}, types: {pass: {uniform: type}}, values}
var mode_order: Array = []
var _log := PackedStringArray()
var _said := {}

var _scene: Node = null
var _pp: CanvasItem = null        # %PostProcessingShader
var _link: CanvasItem = null      # %LinkShader
var _link_copy: CanvasItem = null # %LinkBackBufferCopy
var _orig_pp: Material = null
var _orig_link: Material = null
var _mats := {}                  # mode id -> {pass: ShaderMaterial}
var _ov_copy: BackBufferCopy = null   # the breakup pass: a copy of the screen so far,
var _ov: ColorRect = null             # then the pass itself, after %LinkShader
var _active := ""
var layers := {}                 # id -> {id, name, dir, over, order, shader, types, values}
var layer_order: Array = []
var _lmats := {}                 # layer id -> ShaderMaterial
var _lnodes := {}                # layer id -> [BackBufferCopy, ColorRect]
var _lon: Array = []             # the layers on this frame
var _tinted := false             # the game's camera tint set by a layer
var _chose_checked := false
var _cam: Node = null
var _presets = null
var _t_find := 0.0

var _wrote = null                # what this last put in analog_tuning["signal"]
var _user_signal := false
var _prev_raw := -1.0
var signal_now := -1.0           # for tests and status: the quality last given to the game


func zm_init(_core: Node, _dir: String) -> void:
	_say("signal_sim %s" % VERSION)
	_migrate()
	_read_settings()
	if ResourceLoader.exists(ANALOG_PRESETS):
		_presets = load(ANALOG_PRESETS)
	_load_layers()
	_load_modes()
	var cs = load(DIR + "camera.gd")
	if cs is Script and cs.can_instantiate():
		_cam = cs.new()
		_cam.name = "Camera"
		add_child(_cam)
	_write_status()


# zm_init runs before zonemods puts this in the tree
func _ready() -> void:
	get_tree().node_added.connect(_on_node_added)


## mod_settings: settings.cfg changed in the game
func zm_apply_settings() -> void:
	_say("")
	_say("-- settings changed in the game --")
	_read_settings()
	for id in mode_order:
		modes[id]["values"] = _mode_values(id, modes[id]["dir"])
		if _mats.has(id):
			for ps in _mats[id]:
				_put_values(_mats[id][ps], modes[id]["values"], modes[id]["types"][ps])
		_say(_mode_line(id))
	for id in layer_order:
		layers[id]["values"] = _layer_values(id)
		if _lmats.has(id):
			_put_values(_lmats[id], layers[id]["values"], layers[id]["types"])
		_say(_layer_line(id))
	if _cam != null and _active != "":
		_say(_cam.set_mode(_camera_values(_active)))
	_write_status()


# ---------------------------------------------------------------- settings

func _settings_path() -> String:
	return OS.get_executable_path().get_base_dir().path_join("signal_sim/settings.cfg")


func _read_settings() -> void:
	var c := ConfigFile.new()
	c.load(DIR + "settings.cfg")
	cfg["minimum"] = clampf(_num(c.get_value("signal", "minimum", 0.0)), 0.0, 100.0)
	cfg["intensity"] = clampf(_num(c.get_value("signal", "intensity", 100.0)), 0.0, 200.0)
	if _signal_neutral():
		_say("signal: the game's own (no floor, intensity 100%)")
	else:
		_say("signal: never below %s, walls and range cost %s percent - for every Signal Simulation mode"
			% [_fmt(cfg["minimum"]), _fmt(cfg["intensity"])])


func _signal_neutral() -> bool:
	return cfg["minimum"] <= 0.0 and absf(cfg["intensity"] - 100.0) < 0.001


static func _num(v) -> float:
	if typeof(v) == TYPE_FLOAT or typeof(v) == TYPE_INT:
		return float(v)
	if typeof(v) == TYPE_STRING and str(v).is_valid_float():
		return str(v).to_float()
	return 0.0


static func _fmt(v: float) -> String:
	return str(int(v)) if is_equal_approx(v, roundf(v)) else "%.1f" % v


# analog400mw's settings, once: its [tuning], [link] and [camera] values go into
# [analog_400mw] (the same names), so an upgrade keeps the picture you had
func _migrate() -> void:
	var gd := OS.get_executable_path().get_base_dir()
	var old_p := gd.path_join(OLD_MOD).path_join("settings.cfg")
	var mark := gd.path_join("signal_sim").path_join("migrated.txt")
	if not FileAccess.file_exists(old_p):
		return
	if FileAccess.file_exists(mark):
		_say("%s is still installed: delete its folder, signal_sim does what it did" % OLD_MOD)
		return
	var old := ConfigFile.new()
	if old.load(old_p) != OK:
		return
	var p := _settings_path()
	var text := FileAccess.get_file_as_string(p)
	if text == "":
		return
	var n := 0
	for sec in ["tuning", "link", "camera"]:
		if not old.has_section(sec):
			continue
		for k in old.get_section_keys(sec):
			var r := _set_in_section(text, "analog_400mw", k, var_to_str(old.get_value(sec, k)))
			if r != text:
				text = r
				n += 1
	var f := FileAccess.open(p, FileAccess.WRITE)
	if f != null:
		f.store_string(text)
		f.close()
	var m := FileAccess.open(mark, FileAccess.WRITE)
	if m != null:
		m.store_string("analog400mw's settings were copied into [analog_400mw] of settings.cfg (%d values). "
			% n + "Delete this file to copy them again at the next start.\n")
	_say("moved %d of %s's settings into [analog_400mw]; delete the %s folder" % [n, OLD_MOD, OLD_MOD])


# key=value in a section of a settings file's text, the rest left as it is
static func _set_in_section(text: String, section: String, key: String, value: String) -> String:
	var lines := text.split("\n")
	var inside := false
	var re := RegEx.create_from_string("^\\s*" + key + "\\s*=")
	for i in lines.size():
		var t := lines[i].strip_edges()
		if t.begins_with("[") and t.ends_with("]"):
			inside = t == "[" + section + "]"
			continue
		if inside and re.search(lines[i]) != null:
			var cr := lines[i].ends_with("\r")
			var nl := key + "=" + value + ("\r" if cr else "")
			if nl == lines[i]:
				return text
			lines[i] = nl
			return "\n".join(lines)
	return text


# ---------------------------------------------------------------- modes

# the modes folder by its real path: in the game, res:// lists only what is in
# thezone.pck (loose files next to the exe can be read through it, but its
# folders can't be listed)
func _modes_root() -> String:
	return OS.get_executable_path().get_base_dir().path_join("signal_sim").path_join("modes")


func _load_modes() -> void:
	modes.clear()
	mode_order.clear()
	var root := _modes_root()
	var d := DirAccess.open(root)
	if d == null:
		_say("modes: none (no modes folder)")
		return
	var ident := RegEx.create_from_string("^[a-z0-9_]+$")
	var subs := Array(d.get_directories())
	subs.sort()
	for sub in subs:
		var dir: String = root.path_join(sub) + "/"
		var mc := ConfigFile.new()
		if mc.load(dir + "mode.cfg") != OK:
			continue
		if ident.search(sub) == null:
			_say("SKIP mode %s - its folder name may only have a-z, 0-9 and _" % sub)
			continue
		if not bool(mc.get_value("mode", "enabled", true)):
			_say("off  mode %s (enabled=false in its mode.cfg)" % sub)
			continue
		if layers.has(sub):
			_say("SKIP mode %s - signal_sim 1.0's; it is a layer over the game's analog modes now (layers/%s): delete modes/%s" % [sub, sub, sub])
			continue
		var m := {"id": sub, "dir": dir, "name": str(mc.get_value("mode", "name", sub)),
			"order": int(_num(mc.get_value("mode", "order", 100))),
			"preset": str(mc.get_value("mode", "preset", "")), "pass": {}, "types": {}}
		var bad := ""
		for ps in PASSES:
			var f := str(mc.get_value("mode", ps, ""))
			if f == "":
				if ps == "shader":
					bad = "no shader (mode.cfg: shader=\"<file>.gdshader\")"
				continue
			var path := f if f.begins_with("res://") else dir + f
			var sh := _shader(path)
			if sh == null:
				bad = "%s %s did not compile or is missing: the game's log (Godot's output) has any error" % [ps, f]
				break
			m["pass"][ps] = sh
			m["types"][ps] = _uniforms(sh)
		if bad != "":
			_say("SKIP mode %s - %s" % [sub, bad])
			continue
		if m["preset"] != "" and _preset_values(m["preset"]).is_empty():
			_say("mode %s: the game has no analog preset \"%s\" - its shaders' own defaults instead" % [sub, m["preset"]])
		modes[sub] = m
		m["values"] = _mode_values(sub, dir)
		mode_order.append(sub)
	mode_order.sort_custom(func(a, b):
		if modes[a]["order"] != modes[b]["order"]:
			return modes[a]["order"] < modes[b]["order"]
		return str(modes[a]["name"]) < str(modes[b]["name"]))
	for id in mode_order:
		_say(_mode_line(id))
	if mode_order.is_empty():
		_say("modes: none")


func _mode_line(id: String) -> String:
	var m: Dictionary = modes[id]
	var v: Dictionary = m["values"]
	var used := 0
	for k in v:
		for ps in m["types"]:
			if m["types"][ps].has(k):
				used += 1
				break
	var parts := PackedStringArray()
	if m["pass"].has("link_shader"):
		parts.append("a pass over the OSD")
	if m["pass"].has("overlay_shader"):
		parts.append("a breakup pass")
	if m["preset"] != "":
		parts.append("the game's %s camera values" % m["preset"])
	if _num(v.get("open_range", 0.0)) > 0.0:
		parts.append("gone at about %d m in the open" % roundi(_num(v["open_range"])))
	return "mode %s: \"%s\" - %d value%s%s" % [id, m["name"], used, "" if used == 1 else "s",
		(", " + ", ".join(parts)) if parts.size() > 0 else ""]


# A shader of the mode's own, compiled now, or one of the game's (res://...).
# One that does not compile has no uniforms to list (and the game's log gets
# Godot's error); every mode's has signal_quality or the lens values at least.
func _shader(path: String) -> Shader:
	var sh: Shader = null
	if path.begins_with("res://") and not path.begins_with(DIR):
		if not ResourceLoader.exists(path):
			return null
		sh = load(path) as Shader
	else:
		if not FileAccess.file_exists(path):
			return null
		var code := FileAccess.get_file_as_string(path)
		if RegEx.create_from_string("(?m)^\\s*shader_type\\s+canvas_item\\s*;").search(code) == null:
			_say("%s is not a canvas_item shader (shader_type canvas_item;)" % path)
			return null
		sh = Shader.new()
		sh.code = code
	if sh == null or (sh.get_shader_uniform_list().is_empty() and DisplayServer.get_name() != "headless"):
		return null
	return sh


# the shader's uniforms that take a value, its includes' too: name -> type
static func _uniforms(sh: Shader) -> Dictionary:
	var out := {}
	for u in sh.get_shader_uniform_list():
		match int(u.get("type", 0)):
			TYPE_FLOAT:
				out[u["name"]] = "float"
			TYPE_INT:
				out[u["name"]] = "int"
			TYPE_BOOL:
				out[u["name"]] = "bool"
			TYPE_VECTOR2, TYPE_VECTOR3, TYPE_VECTOR4, TYPE_COLOR:
				out[u["name"]] = "vec"
	return out


func _preset_values(preset: String) -> Dictionary:
	if _presets == null or preset == "":
		return {}
	var all = _presets.get("PRESETS")
	if all is Dictionary and all.has(preset):
		return (all[preset] as Dictionary).duplicate()
	return {}


# the game's preset (live, so the game's own tuning carries through), then
# mode.cfg [settings], then settings.cfg's section named after the mode; then
# the two that stand for others: lines (the analog picture's detail: the
# shaders' resolution, 480 lines = 1) and lens_tint (0 = neutral .. 1 = the
# preset's tint_center/tint_edge)
func _mode_values(id: String, dir: String) -> Dictionary:
	var base := _preset_values(str(modes[id]["preset"])) if modes.has(id) else {}
	var v := base.duplicate()
	var mc := ConfigFile.new()
	if mc.load(dir + "mode.cfg") == OK and mc.has_section("settings"):
		for k in mc.get_section_keys("settings"):
			v[k] = mc.get_value("settings", k)
	var sc := ConfigFile.new()
	if sc.load(DIR + "settings.cfg") == OK and sc.has_section(id):
		for k in sc.get_section_keys(id):
			v[k] = sc.get_value(id, k)
	if _num(v.get("lines", 0.0)) > 0.0:
		v["resolution"] = _num(v["lines"]) / 480.0
	if v.has("lens_tint"):
		var k := clampf(_num(v["lens_tint"]), 0.0, 2.0)
		for tk in ["tint_center", "tint_edge"]:
			var t = base.get(tk, v.get(tk))
			if t is Vector3:
				v[tk] = Vector3.ONE.lerp(t, k)
	return v


func _camera_values(id: String) -> Dictionary:
	var out := {}
	for k in CAMERA_KEYS:
		if modes[id]["values"].has(k):
			out[k] = modes[id]["values"][k]
	return out


static func _put_values(mat: ShaderMaterial, values: Dictionary, types: Dictionary) -> void:
	for k in values:
		if not types.has(k) or k in GAME_UNIFORMS:
			continue
		var v = values[k]
		match types[k]:
			"float":
				mat.set_shader_parameter(k, _num(v))
			"int":
				mat.set_shader_parameter(k, int(_num(v)))
			"bool":
				mat.set_shader_parameter(k, v if typeof(v) == TYPE_BOOL else _num(v) != 0.0)
			_:
				mat.set_shader_parameter(k, v)


# the mode's materials, one per pass it has
func _materials(id: String) -> Dictionary:
	if not _mats.has(id):
		var m: Dictionary = modes[id]
		var out := {}
		for ps in m["pass"]:
			var mat := ShaderMaterial.new()
			mat.shader = m["pass"][ps]
			_put_values(mat, m["values"], m["types"][ps])
			out[ps] = mat
		_mats[id] = out
	return _mats[id]


# ---------------------------------------------------------------- layers

func _layers_root() -> String:
	return OS.get_executable_path().get_base_dir().path_join("signal_sim").path_join("layers")


# Each folder in layers/ with a layer.cfg: a pass drawn on top of the modes
# it is for. Like the modes, a shader that does not compile leaves it out.
func _load_layers() -> void:
	layers.clear()
	layer_order.clear()
	var root := _layers_root()
	var d := DirAccess.open(root)
	if d == null:
		_say("layers: none (no layers folder)")
		return
	var ident := RegEx.create_from_string("^[a-z0-9_]+$")
	var subs := Array(d.get_directories())
	subs.sort()
	for sub in subs:
		var dir: String = root.path_join(sub) + "/"
		var lc := ConfigFile.new()
		if lc.load(dir + "layer.cfg") != OK:
			continue
		if ident.search(sub) == null:
			_say("SKIP layer %s - its folder name may only have a-z, 0-9 and _" % sub)
			continue
		var f := str(lc.get_value("layer", "shader", ""))
		var sh := _shader(f if f.begins_with("res://") else dir + f) if f != "" else null
		if sh == null:
			_say("SKIP layer %s - its shader %s did not compile or is missing: the game's log (Godot's output) has any error" % [sub, f])
			continue
		var over = lc.get_value("layer", "over", "analog")
		layers[sub] = {"id": sub, "dir": dir, "name": str(lc.get_value("layer", "name", sub)),
			"over": str(over) if not (over is Array) else ",".join(PackedStringArray(over)),
			"order": int(_num(lc.get_value("layer", "order", 100))), "shader": sh, "types": _uniforms(sh)}
		layers[sub]["values"] = _layer_values(sub)
		layer_order.append(sub)
	layer_order.sort_custom(func(a, b):
		if layers[a]["order"] != layers[b]["order"]:
			return layers[a]["order"] < layers[b]["order"]
		return a < b)
	for id in layer_order:
		_say(_layer_line(id))
	if layer_order.is_empty():
		_say("layers: none")


# LAYER_DEFAULTS, then the layer's layer.cfg [settings], then settings.cfg's
# section named after it
func _layer_values(id: String) -> Dictionary:
	var v := LAYER_DEFAULTS.duplicate()
	var lc := ConfigFile.new()
	if lc.load(str(layers[id]["dir"]) + "layer.cfg") == OK and lc.has_section("settings"):
		for k in lc.get_section_keys("settings"):
			v[k] = lc.get_value("settings", k)
	var sc := ConfigFile.new()
	if sc.load(DIR + "settings.cfg") == OK and sc.has_section(id):
		for k in sc.get_section_keys(id):
			v[k] = sc.get_value(id, k)
	v["on"] = v["on"] if typeof(v["on"]) == TYPE_BOOL else not (str(v["on"]).to_lower() in ["false", "no", "off", "0"])
	return v


func _layer_line(id: String) -> String:
	var v: Dictionary = layers[id]["values"]
	if not bool(v["on"]):
		return "layer %s: \"%s\" - off (on=false)" % [id, layers[id]["name"]]
	var parts := PackedStringArray(["over %s" % ("the game's analog modes" if layers[id]["over"] == "analog" else layers[id]["over"]),
		"strength %s" % _fmt(_num(v["strength"]))])
	if _num(v["game_noise"]) < 1.0:
		parts.append("the game's link noise at %d percent" % roundi(_num(v["game_noise"]) * 100.0))
	if _num(v["lens_tint"]) < 1.0:
		parts.append("lens tint %s" % ("neutral" if _num(v["lens_tint"]) <= 0.0 else _fmt(_num(v["lens_tint"]))))
	if _num(v["open_range"]) > 0.0:
		parts.append("gone at about %d m in the open" % roundi(_num(v["open_range"])))
	return "layer %s: \"%s\" - %s" % [id, layers[id]["name"], ", ".join(parts)]


func _layer_for(id: String, sim: String, preset: String) -> bool:
	if not bool(layers[id]["values"]["on"]) or _active != "":
		return false
	var over: String = layers[id]["over"]
	if over == "all":
		return sim != "disabled"
	if over == "analog":
		return preset != ""
	return sim in over.split(",")


# Every frame, after the game's physics have set its materials: the layers
# for the mode picked are drawn, the others hidden; the game's link noise
# and lens tint as the first layer on sets them
func _layers_tick() -> void:
	var zs := _zs()
	var sim := str(zs.get("SIGNAL_SIMULATION")) if zs != null else ""
	var preset := str(_presets.preset_for(sim)) if _presets != null else ""
	var on: Array = []
	if _pp != null and is_instance_valid(_pp):
		for id in layer_order:
			if _layer_for(id, sim, preset):
				on.append(id)
	for id in _lnodes.keys():
		if not id in on:
			_layer_show(id, false, null)
	if on.is_empty():
		_untint()
		_lon = on
		return
	var after: CanvasItem = _link if _link != null and _link.visible else _pp
	var q = _pp.material.get_shader_parameter("signal_quality") if _pp.material is ShaderMaterial else null
	var res = _pp.material.get_shader_parameter("resolution") if _pp.material is ShaderMaterial else null
	for id in on:
		var mat := _layer_mat(id)
		if q != null:
			mat.set_shader_parameter("signal_quality", q)
		if res != null and layers[id]["types"].has("resolution"):
			mat.set_shader_parameter("resolution", res)
		after = _layer_show(id, true, after)
	var v: Dictionary = layers[on[0]]["values"]
	# the game's link pass, given part of the signal loss
	var share := clampf(_num(v["game_noise"]), 0.0, 1.0)
	if share < 1.0 and q != null and _link != null and _link.visible and _link.material is ShaderMaterial:
		(_link.material as ShaderMaterial).set_shader_parameter("signal_quality", 100.0 - (100.0 - _num(q)) * share)
	if preset != "":
		_tint(preset, clampf(_num(v["lens_tint"]), 0.0, 1.0))
	_lon = on


func _layer_mat(id: String) -> ShaderMaterial:
	if not _lmats.has(id):
		var mat := ShaderMaterial.new()
		mat.shader = layers[id]["shader"]
		_put_values(mat, layers[id]["values"], layers[id]["types"])
		_lmats[id] = mat
	return _lmats[id]


# The layer's pass: a copy of the screen so far, then the pass, right after
# the node before it, the same size; returns the pass, for the next layer
func _layer_show(id: String, on: bool, after: CanvasItem) -> CanvasItem:
	var pair = _lnodes.get(id)
	if not on:
		if pair != null and is_instance_valid(pair[1]):
			pair[0].visible = false
			pair[1].visible = false
		return null
	var parent := after.get_parent()
	if pair == null or not is_instance_valid(pair[1]) or pair[1].get_parent() != parent:
		var cp := BackBufferCopy.new()
		cp.name = "SignalSimLayerCopy_" + id
		cp.copy_mode = BackBufferCopy.COPY_MODE_VIEWPORT
		var rect := ColorRect.new()
		rect.name = "SignalSimLayer_" + id
		rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
		parent.add_child(cp)
		parent.add_child(rect)
		pair = [cp, rect]
		_lnodes[id] = pair
	var cp: BackBufferCopy = pair[0]
	var rect: ColorRect = pair[1]
	if cp.get_index() != after.get_index() + 1:
		parent.move_child(cp, after.get_index() + 1)
	if rect.get_index() != cp.get_index() + 1:
		parent.move_child(rect, cp.get_index() + 1)
	rect.material = _layer_mat(id)
	if after is Control:
		var c := after as Control
		if rect.position != c.position or rect.size != c.size or rect.scale != c.scale:
			rect.position = c.position
			rect.size = c.size
			rect.scale = c.scale
	cp.visible = true
	rect.visible = true
	return rect


# The game's camera tint, between neutral (k = 0) and the game's own (1),
# set on the game's own material while a layer is on
func _tint(preset: String, k: float) -> void:
	if _pp == null or not (_pp.material is ShaderMaterial) or _ours(_pp.material):
		return
	var zg := _zg()
	var tuning = zg.get("analog_tuning") if zg != null else {}
	var base: Dictionary = _presets.values(preset, tuning if tuning is Dictionary else {})
	var mat := _pp.material as ShaderMaterial
	for tk in ["tint_center", "tint_edge"]:
		var t = base.get(tk)
		if not (t is Vector3):
			continue
		var want: Vector3 = Vector3.ONE.lerp(t, k)
		var have = mat.get_shader_parameter(tk)
		if not (have is Vector3) or not (have as Vector3).is_equal_approx(want):
			mat.set_shader_parameter(tk, want)
	_tinted = k < 1.0


# the game's own tint back, when no layer is on
func _untint() -> void:
	if not _tinted:
		return
	_tinted = false
	var zs := _zs()
	var preset := str(_presets.preset_for(str(zs.get("SIGNAL_SIMULATION")))) if zs != null and _presets != null else ""
	if preset == "" or _pp == null or not is_instance_valid(_pp) or not (_pp.material is ShaderMaterial) or _ours(_pp.material):
		return
	_tint(preset, 1.0)


# signal_sim 1.0 listed Analog 400mw by Nito as a mode of its own: a choice
# of it left from then goes to the game's Analog whoop, which the layer is on
func _check_choice() -> void:
	if _chose_checked:
		return
	var zs := _zs()
	if zs == null:
		return
	_chose_checked = true
	var sim := str(zs.get("SIGNAL_SIMULATION"))
	if sim.begins_with(PREFIX) and layers.has(sim.substr(PREFIX.length())) and not modes.has(sim.substr(PREFIX.length())):
		zs.set("SIGNAL_SIMULATION", "analog_whoop")
		if zs.has_method("save_to_disk"):
			zs.call("save_to_disk")
		_once("choice", "Signal Simulation was \"%s\", a mode in signal_sim 1.0: now the game's Analog whoop, with the layer on top" % layers[sim.substr(PREFIX.length())]["name"])


# ---------------------------------------------------------------- the settings page

func _on_node_added(n: Node) -> void:
	if n is Control and n.get_script() is Script and (n.get_script() as Script).resource_path == SETTINGS_SCRIPT:
		_menu.call_deferred(n)        # after its _ready has filled it in


func _menu(page: Node) -> void:
	if not is_instance_valid(page) or mode_order.is_empty():
		return
	var btn := page.get_node_or_null("%SignalSimulationButton") as OptionButton
	var save := page.get_node_or_null("%SaveButton") as BaseButton
	if btn == null or save == null:
		_once("menu", "menu: the game's settings page has changed - the modes could not be added")
		return
	var zs := _zs()
	var cur := str(zs.get("SIGNAL_SIMULATION")) if zs != null else ""
	var first := btn.item_count
	for id in mode_order:
		btn.add_item(str(modes[id]["name"]))
		btn.set_item_metadata(btn.item_count - 1, PREFIX + id)
		if cur == PREFIX + id:
			btn.select(btn.item_count - 1)
	# what the list shows when Save is pressed: the game's own handler (which
	# runs first) leaves an index it does not know alone, and this sets ours
	var state := {"pick": cur if cur.begins_with(PREFIX) and btn.selected >= first else ""}
	btn.item_selected.connect(func(i: int):
		state["pick"] = str(btn.get_item_metadata(i)) if i >= first else "")
	save.pressed.connect(func():
		var z := _zs()
		if state["pick"] != "" and z != null:
			z.set("SIGNAL_SIMULATION", state["pick"])
			if z.has_method("save_to_disk"):
				z.call("save_to_disk"))
	_once("menu_ok", "menu: %d mode(s) added to Signal Simulation" % mode_order.size())


func _zs() -> Node:
	return get_node_or_null("/root/ZSettings")


func _zg() -> Node:
	return get_node_or_null("/root/ZGamestate")


# ---------------------------------------------------------------- in the game

func _process(delta: float) -> void:
	_check_choice()
	var cs := get_tree().current_scene
	_t_find += delta
	if cs != _scene:
		_leave(true)
		_scene = cs
		_find()
	elif _scene != null and (_pp == null or not is_instance_valid(_pp)) and _t_find > 1.0:
		_find()          # not the in-game scene (yet): look again now and then
	if _pp == null:
		return
	var zs := _zs()
	var sim := str(zs.get("SIGNAL_SIMULATION")) if zs != null else ""
	var want := ""
	if sim.begins_with(PREFIX) and modes.has(sim.substr(PREFIX.length())):
		want = sim.substr(PREFIX.length())
	if want != _active:
		_leave(false)
		_enter(want)
	elif _active != "":
		_keep()
	_layers_tick()


func _find() -> void:
	_t_find = 0.0
	_lnodes.clear()
	_lon = []
	_tinted = false
	_pp = null
	_link = null
	_link_copy = null
	if _scene == null:
		return
	var pp := _scene.get_node_or_null("%PostProcessingShader") as CanvasItem
	if pp == null or not (pp.material is ShaderMaterial):
		return
	_pp = pp
	_link = _scene.get_node_or_null("%LinkShader") as CanvasItem
	_link_copy = _scene.get_node_or_null("%LinkBackBufferCopy") as CanvasItem
	if _link != null and not (_link.material is ShaderMaterial):
		_link = null
	_orig_pp = null
	_orig_link = null
	_active = ""


func _ours(m: Material) -> bool:
	if m == null:
		return false
	for id in _mats:
		for ps in _mats[id]:
			if m == _mats[id][ps]:
				return true
	return false


func _enter(id: String) -> void:
	_active = id
	if id == "":
		if _cam != null:
			_cam.set_mode({})
		return
	var mats := _materials(id)
	if not _ours(_pp.material):
		_orig_pp = _pp.material
	_carry(_orig_pp, mats["shader"])
	_pp.material = mats["shader"]
	if _link != null:
		if not _ours(_link.material):
			_orig_link = _link.material
		if mats.has("link_shader"):
			_link.material = mats["link_shader"]
	if mats.has("overlay_shader"):
		_overlay(true, mats["overlay_shader"])
	_keep()
	if _cam != null:
		_once("cam_" + id, _cam.set_mode(_camera_values(id)))


# every frame while a mode of ours is on: the game sets material.shader and
# shows or hides its link pass when its preset or analog_tuning changes, and
# lays out its nodes again on a resize
func _keep() -> void:
	var mats := _materials(_active)
	var m: Dictionary = modes[_active]
	if _pp.material != mats["shader"]:
		if not _ours(_pp.material):
			_orig_pp = _pp.material
		_pp.material = mats["shader"]
	if mats["shader"].shader != m["pass"]["shader"]:
		mats["shader"].shader = m["pass"]["shader"]
	var q = mats["shader"].get_shader_parameter("signal_quality")
	var has_link := mats.has("link_shader") and _link != null
	if has_link:
		if _link.material != mats["link_shader"]:
			_link.material = mats["link_shader"]
		if mats["link_shader"].shader != m["pass"]["link_shader"]:
			mats["link_shader"].shader = m["pass"]["link_shader"]
		mats["link_shader"].set_shader_parameter("signal_quality", _link_quality(q, m["values"]))
	_show_link(has_link)
	if mats.has("overlay_shader"):
		_overlay(true, mats["overlay_shader"])
		mats["overlay_shader"].set_shader_parameter("signal_quality", q)
	else:
		_overlay(false, null)


# the link pass's quality: link_share (0-1) of the loss, so a mode can keep the
# game's link pass mild and do its breakup in its own pass
static func _link_quality(q, values: Dictionary):
	if q == null or not values.has("link_share"):
		return q
	return 100.0 - (100.0 - _num(q)) * clampf(_num(values["link_share"]), 0.0, 1.0)


func _show_link(on: bool) -> void:
	if _link != null and _link.visible != on:
		_link.visible = on
	if _link_copy != null and _link_copy.visible != on:
		_link_copy.visible = on


# the breakup pass: right after %LinkShader (or the post-process, if the game
# has no link pass), the same size, over everything the link pass covers
func _overlay(on: bool, mat: ShaderMaterial) -> void:
	if not on:
		if _ov != null and is_instance_valid(_ov):
			_ov.visible = false
			_ov_copy.visible = false
		return
	var after: CanvasItem = _link if _link != null else _pp
	if _ov == null or not is_instance_valid(_ov) or _ov.get_parent() != after.get_parent():
		_ov_copy = BackBufferCopy.new()
		_ov_copy.name = "SignalSimBreakupCopy"
		_ov_copy.copy_mode = BackBufferCopy.COPY_MODE_VIEWPORT
		_ov = ColorRect.new()
		_ov.name = "SignalSimBreakup"
		_ov.mouse_filter = Control.MOUSE_FILTER_IGNORE
		var parent := after.get_parent()
		parent.add_child(_ov_copy)
		parent.add_child(_ov)
		parent.move_child(_ov_copy, after.get_index() + 1)
		parent.move_child(_ov, _ov_copy.get_index() + 1)
	if _ov.material != mat:
		_ov.material = mat
	if after is Control:
		var c := after as Control
		if _ov.position != c.position or _ov.size != c.size or _ov.scale != c.scale:
			_ov.position = c.position
			_ov.size = c.size
			_ov.scale = c.scale
	_ov.visible = true
	_ov_copy.visible = true


# back to the game's own materials; the game shows its link pass for its own
# analog modes (it sets that when its preset changes, which may come first)
func _leave(gone: bool) -> void:
	if _active == "" or gone:
		_active = ""
		_ov = null
		_ov_copy = null
		if _cam != null:
			_cam.set_mode({})
		return
	if _pp != null and is_instance_valid(_pp):
		if _ours(_pp.material) and _orig_pp != null:
			_carry(_pp.material, _orig_pp)
			_pp.material = _orig_pp
		if _link != null and is_instance_valid(_link):
			if _ours(_link.material) and _orig_link != null:
				_link.material = _orig_link
			var zs := _zs()
			var sim := str(zs.get("SIGNAL_SIMULATION")) if zs != null else ""
			var game_analog: bool = _presets != null and str(_presets.preset_for(sim)) != ""
			if not game_analog:
				_show_link(false)
		_overlay(false, null)
		# the game may already have set up its new mode on our material (its
		# physics frame came first): have it set up its own again, on its own
		# material - shader, link pass and preset values
		if _scene != null and is_instance_valid(_scene) and "_post_shader_preset" in _scene:
			_scene.set("_post_shader_preset", "-")
	_active = ""
	if _cam != null:
		_cam.set_mode({})


static func _carry(from: Material, to: Material) -> void:
	if not (from is ShaderMaterial) or not (to is ShaderMaterial):
		return
	for k in GAME_UNIFORMS:
		var v = from.get_shader_parameter(k)
		if v != null:
			to.set_shader_parameter(k, v)


# ---------------------------------------------------------------- the signal

func _physics_process(_delta: float) -> void:
	var zg := _zg()
	if zg == null:
		return
	var tuning = zg.get("analog_tuning")
	if not (tuning is Dictionary):
		return
	# the player's own /analogtune signal wins until they reset it
	if tuning.has("signal") and (_wrote == null or not is_equal_approx(_num(tuning["signal"]), _num(_wrote))):
		if not _user_signal:
			_once("user", "signal: set with /analogtune signal - left as you set it until /analogtune reset")
		_user_signal = true
		_wrote = null
		return
	_user_signal = false
	var k := _open_loss()
	var zs := _zs()
	var sim := str(zs.get("SIGNAL_SIMULATION")) if zs != null else "disabled"
	var main := _scene if _pp != null else null
	if (_signal_neutral() and k <= 0.0) or sim == "disabled" or main == null or main.get("_last_signal_quality") == null:
		if _wrote != null:
			tuning.erase("signal")
			_wrote = null
		_prev_raw = -1.0
		signal_now = -1.0
		return
	# the game's own quality, as it works it out: the better of this frame's
	# and the last frame's rays (they alternate between two spots), and a
	# map's multipath probe at half strength when the signal is bad
	var raw := _num(main.get("_last_signal_quality"))
	var q := maxf(maxf(raw, _prev_raw), 0.1)
	_prev_raw = raw
	if q < 50.0 and main.has_method("_calculate_signal_quality"):
		var probe := main.get_node_or_null("Map/MultipathProbe") as Node3D
		if probe != null:
			q = maxf(q, _num(main.call("_calculate_signal_quality", probe.global_position)) * 0.5)
	if k > 0.0:
		q -= k * _range_m(zg)
	q = 100.0 - (100.0 - q) * cfg["intensity"] / 100.0
	q = clampf(maxf(q, cfg["minimum"]), 0.0, 100.0)
	tuning["signal"] = q
	_wrote = q
	signal_now = q


# the active mode's (or the layer on's) open-air loss, quality points per metre: the signal is gone
# (quality 0) at its open_range in the open
func _open_loss() -> float:
	var r := 0.0
	if _active != "" and modes.has(_active):
		r = _num(modes[_active]["values"].get("open_range", 0.0))
	elif not _lon.is_empty() and layers.has(_lon[0]):
		r = _num(layers[_lon[0]]["values"].get("open_range", 0.0))
	return 100.0 / r if r > 0.0 else 0.0


# metres from the pilot (at the spawn, where the game puts the receiver) to the quad's antenna
func _range_m(zg: Node) -> float:
	var zc := get_node_or_null("/root/ZCommands")
	var player = zc.get("player") if zc != null else null
	var rt = zg.get("respawn_transform")
	if not (player is Node3D) or not (rt is Transform3D):
		return 0.0
	var ant := (player as Node3D).get_node_or_null("Vehicle/VTXAntenna") as Node3D
	var at: Vector3 = ant.global_position if ant != null else (player as Node3D).global_position
	return (rt.origin + Vector3(0, 2, 0)).distance_to(at)


# ---------------------------------------------------------------- status

func _once(key: String, s: String) -> void:
	if _said.has(key):
		return
	_said[key] = true
	_say(s)
	_write_status()


func _say(s: String) -> void:
	_log.append(s)
	print("[signal_sim] " + s)


func _write_status() -> void:
	var p := OS.get_executable_path().get_base_dir().path_join("signal_sim/status.txt")
	var f := FileAccess.open(p, FileAccess.WRITE)
	if f != null:
		f.store_string("\n".join(_log) + "\n")
