signal_sim 1.1
==================

Signal Simulation, opened up: "Analog 400mw by Nito", a 400 mW whoop's
analog breakup drawn on top of the game's own analog modes; for every mode,
a floor under the signal and how much walls and range cost, so you can fly
further and still see; and room for layers and modes of your own, each a
shader in a folder.


ANALOG 400MW BY NITO
  Analog 5.8 GHz FPV video from a 400 mW analog whoop (Meteor75 Pro II
  class), as extra taste on the game's own analog modes: pick Analog 2500 or
  Analog whoop in Pause > Game Settings > Signal Simulation and it is drawn
  on top, after the game's camera and link passes:
  * close in, every so often a colour hit: for a frame, below one line,
    coloured things swing through the wrong hues and back, in bursts
  * on a strong signal no noise of its own; with range and walls, white
    specks and short white streaks, more and more
  * breakup bands: a stretch of lines where the colour locks wrong for a few
    frames, one hue across it, patchy; on a weak signal with white sparkle
    through it
  * far out the colour fades and flickers, and at the end the picture sinks
    into white snow
  It goes over the OSD too, as on real goggles. It also keeps only part of
  the game's own link noise (its colourful speckle - DVR noise is mostly
  white), sets the game's camera tint to neutral, and fades the signal in
  the open: gone at about 400 m (walls cut that much shorter).

  Its settings, in settings.cfg [analog_400mw] or Pause > Mods (F9 while
  flying), change as you move a slider:
    on           false = the game's analog modes as the game has them
    strength     how much breakup, 0 to 2 (1 = as tuned to DVR footage)
    game_noise   how much of the game's own link noise is kept, 0 to 1
    lens_tint    the camera's colour cast, 0 = neutral to 1 = the game's
    open_range   metres the video reaches in the open; 0 = the game's way

  In signal_sim 1.0 this was a Signal Simulation mode of its own. If that
  was picked, the game's Analog whoop is picked instead at the next start.


THE SIGNAL, FOR EVERY MODE
  settings.cfg [signal] (or Pause > Mods > signal_sim):

  minimum     The weakest the signal gets, 0 to 100. At 60 the picture never
              gets worse than at 60 quality, so you can fly far behind walls
              and keep a picture, with some breakup. 0 = no floor.
  intensity   How much walls and range cost, in percent. 50 = half the loss
              at the same spot, 200 = twice. 100 = the game's own.

  These work on all of them: the game's digital modes (their blockiness and
  frame drops), its analog modes and the modes here. Left at 0 and 100,
  signal_sim leaves the game's signal alone.

  The game's chat command /analogtune signal <0-100> (in its own analog
  modes) still wins: while you have set the signal with it, signal_sim leaves
  it alone. /analogtune reset hands it back.


INSTALL
  1. Steam: right-click The Zone FPV > Manage > Browse local files.
  2. Extract the zip into that folder, so override.cfg and the zonemods and
     signal_sim folders sit next to thezone.exe. Replace files when asked -
     your settings are kept (see UPDATING in zonemods/README.txt).
  3. In game: Pause > Game Settings > Signal Simulation > Analog whoop (or
     Analog 2500) > Save.


UPGRADING FROM ANALOG400MW
  signal_sim replaces analog400mw, which the game's October 2026 update
  switched off (the game now has analog video of its own, and changed the
  files analog400mw replaced). Delete the analog400mw folder. At its first
  launch signal_sim copies the analog400mw settings it still has a place
  for (open_range) into [analog_400mw]; signal_sim/migrated.txt records
  that. Then pick Analog whoop in the game's settings once.


LAYERS
  A layer is a folder in signal_sim/layers/ with a layer.cfg and a shader,
  drawn on top of the modes it is for. layers/analog_400mw is one to copy.

  layer.cfg:
    [layer]
    name="My Layer"             its name, in status.txt
    over="analog"               the game's analog modes; "all" = every mode
                                but Disabled; or modes by name:
                                "analog_whoop,digital_vista_700"
    shader="my_layer.gdshader"  in the folder, or one of the game's (res://)
    order=100                   with more than one, lower is drawn first

    [settings]
    my_value=0.5                a value for the shader's uniform my_value

  Its section in settings.cfg (named after its folder) can set on,
  strength, game_noise, lens_tint and open_range, as above, and any of its
  shader's uniforms. The shader is a canvas_item shader reading the screen
  so far, as a mode's link pass does (see below); signal_quality is set
  every frame, and resolution too if it has one (the game's analog presets'
  detail).


MAKING A MODE
  A mode is a folder in signal_sim/modes/ (letters a-z, digits and _ in its
  name) with a mode.cfg and a shader. At the next start it is in the Signal
  Simulation list, after the game's. None ship with signal_sim.

  mode.cfg:
    [mode]
    name="My Mode"              the name in the list
    shader="my_mode.gdshader"   drawn over the 3D view, before the OSD
    preset=""                   optional: one of the game's analog presets
                                ("whoop", "good") as the starting values
    link_shader=""              optional: a pass over the picture and the
                                OSD, as the game's analog link pass is
    overlay_shader=""           optional: one more pass after that
    order=100                   where in the list (lower = earlier)
    enabled=true                false leaves it out

  A shader is a file in the mode's folder, or one of the game's own by its
  res:// path, e.g.
    shader="res://ingame/analog_v2.gdshader"        the game's analog camera
    link_shader="res://ingame/analog_link.gdshader" the game's link pass

    [settings]
    my_value=0.5                a value for any pass's uniform my_value
    open_range=0                metres: the signal also fades in the open,
                                quality 0 at this range
    link_share=1                how much of the signal loss the link pass
                                gets, 0 to 1 (the others get all of it)
    lines=0                     for the game's analog shaders: the picture's
                                detail in lines (their resolution, 480 = 1)
    lens_tint=1                 for a preset: 0 = neutral, 1 = its tint
    bloom=0, bloom_size=1       the map's glow (Forward+ only)

  A mode's own shader: a canvas_item shader, as the game's post-process is.
    shader_type canvas_item;
    #include "res://ingame/fisheye.gdshaderinc"     // the game's lens
    uniform sampler2D screen_tex : hint_screen_texture;
    uniform float signal_quality = 100.0;  // 0-100, set every frame
    uniform float my_value = 0.5;
    void fragment() {
        vec3 c = texture(screen_tex, fisheye(SCREEN_UV)).rgb;
        float lost = 1.0 - signal_quality / 100.0;
        COLOR = vec4(mix(c, vec3(dot(c, vec3(0.3, 0.59, 0.11))), lost * my_value), 1.0);
    }
  The include gives fisheye(uv) and its two uniforms (fisheye_strength,
  fisheye_crop), which the game sets from the drone's camera settings: use
  it in the first pass, or the picture won't match the game's lens. In a
  link or overlay pass, read the screen without fisheye (it is already in
  the picture); signal_quality is set there too. The game's
  ingame/analog_common.gdshaderinc (noise, YUV, resolution) can be included
  as well. Uniforms of type float, int, bool and vec2/3/4 take values from
  [settings], and from a section of settings.cfg named after the mode.

  A mode's [settings] are read at each start. A mode or layer whose shader
  does not compile is left out: status.txt says so, and the game's log
  (Godot's output) has the error. While a mode of yours is on, the layers
  are not drawn.


UNINSTALL
  To turn signal_sim off without uninstalling it, set enabled=false at the top
  of signal_sim/settings.cfg. To remove it, delete the signal_sim folder; if
  no other mods use zonemods, delete override.cfg and the zonemods folder
  too. If a mode of your own was picked, pick another Signal Simulation.


TROUBLESHOOTING
  signal_sim/status.txt is rewritten at every launch: the layers and modes
  it found, the signal settings, and whether any modes went into the
  settings page. zonemods/status.txt lists which mods loaded.

  No breakup on the game's analog modes
      on=false or strength=0 in [analog_400mw], or a mode of your own
      is picked (layers are not drawn on those). status.txt says which.

  "Analog 400mw by Nito" is still in the Signal Simulation list
      An old signal_sim 1.0 folder, modes/analog_400mw, is still there: it
      is left out (status.txt says so), but delete it.

  Your own mode is not in the list
      status.txt says why: left out (its shader did not compile, or
      mode.cfg names no shader), or the settings page changed in a game
      update.

  No bloom in a mode of your own
      On OpenGL (the Compatibility renderer, which the game falls back to
      without Vulkan) a mode's bloom is left out: it washes the picture out
      white there. status.txt says so.


WHAT IT CHANGES
  No file of the game's is replaced. While the game runs:
  * in the game's analog modes, a layer's pass goes in after the game's link
    pass (a copy of the screen, then the pass, the same size); the game's
    link pass is given game_noise of the signal loss and its camera
    material the lens tint; both go back when the layer goes off
  * with modes of your own: the Signal Simulation list in the settings page
    gets them as extra entries (picking one stores "signal_sim:<mode>" in
    the game's settings); while one is on, the game's post-process node
    draws with the mode's own material, its link pass with the mode's
    link_shader (or stays off), and a pass of signal_sim's own comes after
    that if the mode has an overlay_shader; the game's own come back with
    any other mode
  * with a floor, an intensity or an open_range set, the signal quality
    the game works out is changed and handed back to the game through its
    own analog tuning (what /analogtune signal sets)
