extends Node
## player_list 1.0
##
## The game's list of the other players in the lobby (top left: each name,
## with the buttons to watch them in a small window or full screen) kept out
## of the picture: hidden, or tucked away until the mouse goes to the left
## edge of the screen, when it slides out over a dark panel and back once the
## mouse moves off it. And, in the chat, who joins and leaves the lobby -
## with who was already there when you arrive.
##
## How: the game's in-game scene (ingame_main) keeps the list in
## CanvasLayer/LeftSideBox and the players it hears from in online_players,
## a player dropped 5 s after it last hears from them. This moves and hides
## that box, and compares online_players from frame to frame, putting what
## changed into the chat with the game's own add_chat_message. Nothing of the
## game's is replaced.
##
## Uninstall: delete the player_list folder.

const VERSION := "1.0"
const DIR := "res://player_list/"
const DEFAULTS := {"player_list": "flyout", "join_leave_messages": true}
const MODES := ["flyout", "hidden", "shown"]

const HOME := Vector2(16, 16)       # where the game puts the list (ingame_main _set_viewport_size)
const EDGE_OF_HEIGHT := 0.025       # the hover strip at the left edge: 27 px at 1080p
const SLIDE_S := 0.16               # sliding out or back
const CLOSE_AFTER_S := 0.35         # with the mouse off it
const PAD := 8.0                    # the panel round the list
const MSG_COLOUR := Color(1.0, 0.85, 0.45)
const BATCH_S := 1.0                # joins or leaves this close together go in one message
const ARRIVE_S := 3.0               # players heard from this soon after you arrive were already there
const ARRIVE_MAX_S := 25.0
const REJOIN_S := 120.0             # back within this long: "rejoined"
const NAMES_MAX := 3                # more players than this in a message: how many, not who

var cfg := {}
var _log := PackedStringArray()
var _now := 0.0

var _scene: Node = null
var _main: Node = null               # the in-game scene, when it is one
var _box: Control = null             # its LeftSideBox
var _scroll: Control = null          # %PlayerListContainerContainer
var _list: Control = null            # %PlayerListContainer
var _panel: Panel = null
var _empty: Label = null
var _applied := ""                   # the mode last put on the list
var _open := 0.0                     # flyout: 0 tucked away .. 1 out
var _want_open := false
var _away := 0.0
var _mouse := Vector2(-1, -1)        # where the mouse is, in the window's picture
var _mouse_in := true
var _window: Window = null

var _known := {}                     # user id -> name, of the players in online_players
var _arriving := true
var _attach_t := 0.0
var _loaded_t := -1.0                # when the loading screen went
var _login_t := -1.0                 # when the game logged in to its server (newer versions)
var _gone := {}                      # user id -> [name, when], the players who left
var _batch := {"joined": [], "left": [], "rejoined": []}
var _batch_t := -1.0


func zm_init(_core: Node, _dir: String) -> void:
	_say("player_list %s" % VERSION)
	_read_settings()
	_write_status()


func _read_settings() -> void:
	cfg = DEFAULTS.duplicate()
	var c := ConfigFile.new()
	if c.load(DIR + "settings.cfg") == OK and c.has_section("player_list"):
		for k in c.get_section_keys("player_list"):
			if cfg.has(k):
				cfg[k] = c.get_value("player_list", k)
	var m := str(cfg["player_list"]).strip_edges().to_lower()
	cfg["player_list"] = m if m in MODES else "flyout"
	cfg["join_leave_messages"] = str(cfg["join_leave_messages"]).strip_edges().to_lower() in ["true", "1", "yes", "on"]
	_say("player list: %s; join and leave messages in the chat: %s" % [
		{"flyout": "slides out from the left edge of the screen with the mouse there",
		 "hidden": "hidden", "shown": "as the game shows it"}[cfg["player_list"]],
		"yes" if cfg["join_leave_messages"] else "no"])


## Called by mod_settings when settings.cfg has been changed in the game
func zm_apply_settings() -> void:
	_read_settings()
	_write_status()


func _process(delta: float) -> void:
	_now += delta
	var cs := get_tree().current_scene
	if cs != _scene:
		_scene = cs
		_attach()
	if _main == null or not is_instance_valid(_main):
		_main = null
		return
	_place_list(delta)
	_watch_players()


# The mouse, as the game's window has it
func _input(event: InputEvent) -> void:
	if event is InputEventMouse:
		_mouse = (event as InputEventMouse).position
		_mouse_in = true


func _attach() -> void:
	_detach()
	_main = null
	if _scene == null or not ("online_players" in _scene) or not _scene.has_method("add_chat_message"):
		return
	_main = _scene
	_box = _scene.get_node_or_null("CanvasLayer/LeftSideBox") as Control
	_scroll = _scene.get_node_or_null("%PlayerListContainerContainer") as Control
	_list = _scene.get_node_or_null("%PlayerListContainer") as Control
	if _box == null or _list == null:
		_say("in-game: the player list is not where this looks for it - left as it is")
		_box = null
	_known.clear()
	_gone.clear()
	for k in _batch:
		_batch[k] = []
	_batch_t = -1.0
	_arriving = true
	_attach_t = _now
	_loaded_t = -1.0
	_login_t = -1.0
	_applied = ""
	_open = 0.0
	_want_open = false
	_away = 0.0
	var w := get_viewport() as Window
	if w != _window:
		_window = w
		if _window != null and not _window.mouse_exited.is_connected(_mouse_left):
			_window.mouse_exited.connect(_mouse_left)


func _mouse_left() -> void:
	_mouse_in = false


# The game's list put back as the game has it, and the panel gone
func _detach() -> void:
	if _box != null and is_instance_valid(_box):
		_box.visible = true
		_box.position = HOME
	if _panel != null and is_instance_valid(_panel):
		_panel.queue_free()
	_panel = null
	_empty = null
	_box = null


func _exit_tree() -> void:
	_detach()


# ---- the list --------------------------------------------------------------

func _place_list(delta: float) -> void:
	if _box == null or not is_instance_valid(_box) or _list == null or not is_instance_valid(_list):
		_box = null
		return
	var mode := str(cfg["player_list"])
	if mode != _applied:
		_applied = mode
		_open = 0.0
		_want_open = false
		_box.visible = mode == "shown"
		_box.position = HOME
		if _panel != null and is_instance_valid(_panel):
			_panel.visible = false
	if mode == "shown":
		return
	if mode == "hidden":
		_box.visible = false
		return
	_flyout(delta)


func _flyout(delta: float) -> void:
	var ws := _box.get_viewport_rect().size
	var gs := get_node_or_null("/root/ZGamestate")
	var paused: bool = gs != null and gs.get("is_paused") == true
	# the game hides the pointer after 3 s without the mouse moving
	var pointer: bool = _mouse_in and _mouse.x >= 0.0 and Input.mouse_mode == Input.MOUSE_MODE_VISIBLE and not paused
	var list_h := _scroll.size.y if _scroll != null and is_instance_valid(_scroll) and _scroll.size.y > 0.0 else ws.y * 0.65
	var edge := maxf(ws.y * EDGE_OF_HEIGHT, 6.0)
	var at_edge: bool = pointer and _mouse.x <= edge and _mouse.y <= HOME.y + list_h + PAD
	var on_it: bool = pointer and _open > 0.0 and _panel != null and is_instance_valid(_panel) and _panel.visible \
		and _panel.get_global_rect().grow(24.0).has_point(_mouse)
	if _want_open:
		if at_edge or on_it:
			_away = 0.0
		else:
			_away += delta
			if _away >= CLOSE_AFTER_S or not pointer:
				_want_open = false
	elif at_edge:
		_want_open = true
		_away = 0.0
	_open = move_toward(_open, 1.0 if _want_open else 0.0, delta / SLIDE_S)
	if _open <= 0.0:
		_box.visible = false
		if _panel != null and is_instance_valid(_panel):
			_panel.visible = false
		return
	_make_panel()
	var e := 1.0 - pow(1.0 - _open, 3.0)
	var shown := _shown_items()
	var w := maxf(_list.size.x, _scroll.size.x if _scroll != null and is_instance_valid(_scroll) else 0.0)
	var h := minf(_list.get_combined_minimum_size().y, list_h) if shown > 0 else _empty.get_combined_minimum_size().y
	w = maxf(w, _empty.get_combined_minimum_size().x if shown == 0 else 0.0)
	var closed_x := -(w + HOME.x + PAD * 2.0)
	var x := lerpf(closed_x, HOME.x, e)
	_box.visible = true
	_box.position = Vector2(roundf(x), HOME.y)
	_panel.visible = true
	_panel.position = Vector2(roundf(x) - PAD, HOME.y - PAD)
	_panel.size = Vector2(w + PAD * 2.0, h + PAD * 2.0)
	_empty.visible = shown == 0
	_empty.position = Vector2(PAD, PAD)


func _shown_items() -> int:
	var n := 0
	for c in _list.get_children():
		if c is CanvasItem and (c as CanvasItem).visible and not c.is_queued_for_deletion():
			n += 1
	return n


# A dark panel behind the list while it is out, for it to read over the picture
func _make_panel() -> void:
	if _panel != null and is_instance_valid(_panel):
		return
	_panel = Panel.new()
	_panel.name = "ZmPlayerListPanel"
	_panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var sb := StyleBoxFlat.new()
	sb.bg_color = Color(0.03, 0.04, 0.06, 0.72)
	sb.corner_radius_top_right = 6
	sb.corner_radius_bottom_right = 6
	sb.corner_radius_top_left = 6
	sb.corner_radius_bottom_left = 6
	_panel.add_theme_stylebox_override("panel", sb)
	_empty = Label.new()
	_empty.text = "No other players here"
	_empty.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_empty.add_theme_font_size_override("font_size", 18)
	_empty.add_theme_color_override("font_color", Color(1, 1, 1, 0.7))
	_panel.add_child(_empty)
	var parent := _box.get_parent()
	parent.add_child(_panel)
	parent.move_child(_panel, _box.get_index())


# ---- joining and leaving -------------------------------------------------

func _watch_players() -> void:
	var players = _main.get("online_players")
	if not (players is Dictionary):
		return
	if _arriving:
		_arrival(players)
		return
	var now_ids := {}
	for id in players:
		now_ids[id] = true
		if not _known.has(id):
			var nm := _name_of(players[id])
			_known[id] = nm
			if _batch["left"].has(nm) and _gone.has(id) and _now - float(_gone[id][1]) < BATCH_S * 2.0:
				# gone and back before it was said: nothing to say
				_batch["left"].erase(nm)
				_gone.erase(id)
			elif _gone.has(id) and _now - float(_gone[id][1]) < REJOIN_S:
				_gone.erase(id)
				_add("rejoined", nm)
			else:
				_gone.erase(id)
				_add("joined", nm)
		else:
			# a name the game has only now (or a new one)
			var nm2 := _name_of(players[id])
			if nm2 != "" and nm2 != str(_known[id]):
				_known[id] = nm2
	for id in _known.keys():
		if not now_ids.has(id):
			var nm := str(_known[id])
			_known.erase(id)
			_gone[id] = [nm, _now]
			if _batch["joined"].has(nm):
				_batch["joined"].erase(nm)      # in and out before it was said
			else:
				_add("left", nm)
	if _batch_t >= 0.0 and _now - _batch_t >= BATCH_S:
		_flush()
	if _gone.size() > 64:
		for id in _gone.keys():
			if _now - float(_gone[id][1]) > REJOIN_S:
				_gone.erase(id)


# Players the game hears from just after you arrive were there already: one
# message saying who, once the game has been in the lobby long enough to
# hear from everyone (it sends its status every second)
func _arrival(players: Dictionary) -> void:
	_known.clear()
	for id in players:
		_known[id] = _name_of(players[id])
	var ls := _main.get_node_or_null("LoadingScreen") as CanvasItem
	if _loaded_t < 0.0 and (ls == null or not ls.visible):
		_loaded_t = _now
	var mp = _main.get("mp")
	var uid = mp.get("user_id") if mp is Object else null
	if _login_t < 0.0 and uid != null and str(uid) != "":
		_login_t = _now
	var settled := _loaded_t >= 0.0 and _now - _loaded_t >= ARRIVE_S \
		and (uid == null or (_login_t >= 0.0 and _now - _login_t >= ARRIVE_S))
	if not settled and _now - _attach_t < ARRIVE_MAX_S:
		return
	_arriving = false
	if not bool(cfg["join_leave_messages"]) or _known.is_empty():
		return
	var names := []
	for id in _known:
		names.append(str(_known[id]))
	names.sort_custom(func(a, b): return a.naturalnocasecmp_to(b) < 0)
	if names.size() <= NAMES_MAX:
		_chat("In this lobby: %s" % _names(names))
	else:
		_chat("%d other players in this lobby" % names.size())


func _name_of(p) -> String:
	var st = p.get("status") if p is Dictionary else null
	var nm = st.get("display_name", "") if st is Dictionary else ""
	nm = str(nm).strip_edges() if nm != null else ""
	return nm if nm != "" else "A player"


func _add(kind: String, nm: String) -> void:
	_batch[kind].append(nm)
	if _batch_t < 0.0:
		_batch_t = _now


func _flush() -> void:
	_batch_t = -1.0
	for kind in ["left", "joined", "rejoined"]:
		var names: Array = _batch[kind]
		_batch[kind] = []
		if names.is_empty() or not bool(cfg["join_leave_messages"]):
			continue
		if names.size() <= NAMES_MAX:
			_chat("%s %s" % [_names(names), kind])
		else:
			_chat("%d players %s" % [names.size(), kind])


static func _names(a: Array) -> String:
	if a.size() == 1:
		return str(a[0])
	return "%s and %s" % [", ".join(PackedStringArray(a.slice(0, a.size() - 1))), str(a[-1])]


# A line in the game's chat, in a colour of its own so it reads apart from
# what players say
func _chat(text: String) -> void:
	if _main == null or not is_instance_valid(_main):
		return
	_main.call("add_chat_message", text)
	var box := _main.get_node_or_null("CanvasLayer/ChatMarginContainer/ChatVBox/ScrollContainer/ChatMessagesContainer")
	if box != null and box.get_child_count() > 0:
		var l := box.get_child(box.get_child_count() - 1)
		if l is Label and (l as Label).text == text:
			(l as Label).add_theme_color_override("font_color", MSG_COLOUR)


func _say(s: String) -> void:
	_log.append(s)
	if _log.size() > 200:
		_log = _log.slice(_log.size() - 150)
	print("[player_list] " + s)


func _write_status() -> void:
	var f := FileAccess.open(OS.get_executable_path().get_base_dir().path_join("player_list/status.txt"), FileAccess.WRITE)
	if f != null:
		f.store_string("\n".join(_log) + "\n")
