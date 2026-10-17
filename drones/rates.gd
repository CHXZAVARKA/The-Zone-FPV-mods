extends Node
## drones: rates per drone (settings.cfg [rates] per_drone=true).
##
## The game has one set of rates (ZSettings.RATES), whichever drone is
## flown. With per_drone on, each drone keeps its own: rates changed while a
## drone is picked (the game's Rates page, or anything else that sets them)
## are kept for that drone in drones/rates.cfg, and picking a drone puts its
## rates back. A drone picked for the first time keeps the rates in effect,
## and from then on has its own. The game's drones and the ones this mod adds
## each count as a drone of their own (ZGamestate.selected_vehicle_id).
##
## Checked four times a second: the game reads ZSettings.RATES as it flies, so
## a swap takes effect at once. Nothing is written into the game's own
## settings file; it keeps saving the rates in effect as it always has.

const CHECK_S := 0.25
const AXES := ["roll", "pitch", "yaw"]

var on := false
var file := ""
var store := {}                  # vehicle id -> the snapshot kept for it
var _vid := ""                   # the drone whose rates are in effect
var _last := []                  # the rates in effect, as last seen
var _t := 0.0
var swaps := 0                   # for tests and status: rates put back on a drone change
var saves := 0                   # for tests and status: rates kept for a drone


## Reads [rates] per_drone from the settings and rates.cfg. -> a line for status.txt
func setup(settings_paths: Array, rates_file: String) -> String:
	var want := false
	for p in settings_paths:
		var c := ConfigFile.new()
		if c.load(p) == OK and c.has_section_key("rates", "per_drone"):
			var v = c.get_value("rates", "per_drone")
			want = v if v is bool else str(v).strip_edges().to_lower() in ["true", "yes", "on", "1"]
	file = rates_file
	store = {}
	var rc := ConfigFile.new()
	if rc.load(file) == OK:
		for id in rc.get_sections():
			var s := _from_cfg(rc, id)
			if not s.is_empty():
				store[id] = s
	if want and not on:
		_vid = ""                # look again at the drone picked
	on = want
	set_process(on)
	if not on:
		return "rates: one set for every drone, as the game has it (per_drone=false)"
	return "rates per drone: on - %d drone(s) with rates of their own in rates.cfg; a drone picked for the first time keeps the rates in effect" % store.size()


func _process(dt: float) -> void:
	if not on:                   # set_process(false) before the node is in the tree does not hold
		return
	_t += dt
	if _t < CHECK_S:
		return
	_t = 0.0
	var zs := get_node_or_null("/root/ZSettings")
	var gs := get_node_or_null("/root/ZGamestate")
	if zs == null or gs == null:
		return
	var r = zs.get("RATES")
	var v = gs.get("selected_vehicle_id")
	if not (r is Dictionary) or v == null or str(v) == "":
		return
	var vid := str(v)
	var cur := _snap(r)
	if cur.is_empty():
		return
	if vid == _vid:
		if cur != _last:
			_keep(vid, cur)
			_last = cur
		return
	# another drone picked: whatever changed for the last one is kept first
	if _vid != "" and cur != _last:
		_keep(_vid, cur)
	_vid = vid
	if store.has(vid):
		if store[vid] != cur:
			zs.set("RATES", _to_rates(store[vid], r))
			swaps += 1
		_last = _snap(zs.get("RATES"))
	else:
		_keep(vid, cur)
		_last = cur


# [type, roll x3, pitch x3, yaw x3], rounded, or [] if the rates are not as expected
func _snap(r: Dictionary) -> Array:
	var out := [str(r.get("type", "actual"))]
	for a in AXES:
		var v = r.get(a)
		if not (v is Vector3):
			return []
		out.append_array([snappedf(v.x, 0.0001), snappedf(v.y, 0.0001), snappedf(v.z, 0.0001)])
	return out


# A new RATES dictionary from a snapshot, keeping any other keys the game has
func _to_rates(s: Array, was: Dictionary) -> Dictionary:
	var d := was.duplicate(true)
	d["type"] = s[0]
	for i in AXES.size():
		d[AXES[i]] = Vector3(s[1 + 3 * i], s[2 + 3 * i], s[3 + 3 * i])
	return d


func _keep(vid: String, s: Array) -> void:
	store[vid] = s
	saves += 1
	var rc := ConfigFile.new()
	rc.load(file)
	rc.set_value(vid, "type", s[0])
	for i in AXES.size():
		rc.set_value(vid, AXES[i], Vector3(s[1 + 3 * i], s[2 + 3 * i], s[3 + 3 * i]))
	rc.save(file)


func _from_cfg(rc: ConfigFile, id: String) -> Array:
	var r := {"type": str(rc.get_value(id, "type", "actual"))}
	for a in AXES:
		var v = rc.get_value(id, a, null)
		if not (v is Vector3):
			return []
		r[a] = v
	return _snap(r)
