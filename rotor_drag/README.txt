rotor_drag 1.2
===============

Spinning props in moving air. Adds the drag they put on a quad moving
across them, which The Zone FPV leaves out, and takes off a drag it should
not have: the physics engine's own damping. Let go at course speed, a
whoop now stops in about two thirds of the distance it slid before, and
falls and long coasts carry the weight they should. And in a climb the
props keep their thrust as real ones do, so a punch-out keeps pulling
instead of fading after the first quarter of a second.


WHAT IT CHANGES
  The game slows a quad with air resistance on its body, which grows with
  the square of the speed, and with the physics engine's damping. At the
  few metres a second of a whoop course the air resistance is tiny: the
  game's 65mm Whoop, flown level at 14 km/h and let go, coasts about 6 m
  before it is down to half speed, and 13 m before it is nearly stopped -
  and a third of what stops it is the damping.

  That damping is no air at all. Godot, the engine the game is made in,
  slows every moving body by a tenth of its speed each second (its
  default_linear_damp, 0.1) unless a game sets it otherwise, and The Zone
  FPV leaves it: falling, climbing or coasting at any speed, the quad loses
  10% of its speed a second on top of the air's drag, as if it were lighter
  than it is. rotor_drag takes it off the quad you fly (engine_damping=0); a 5"
  falling flat then reaches 72 km/h rather than 65, and coasts further.

  A real quad also has rotor drag. Its props draw air in and throw it out
  along their axis, so air arriving across them - the quad moving sideways
  or forwards, or wind - has that motion taken out of it, and the quad is
  pushed back. That force grows with the speed itself, not its square, so it
  does most of the slowing at low speeds, and it grows with how much air the
  props move. Ducts turn more of that air along the axis, so a whoop's is
  about twice an open-prop quad's of the same size.

  With rotor_drag - rotor drag on, the engine's damping off - from 14 km/h,
  flown level and let go (measured in Godot with the game's flight model
  and physics settings):

    quad             half speed after     nearly stopped (2 km/h) after
                     game   rotor_drag    game   rotor_drag
    65mm Whoop       6.1 m  3.9 m         13 m   7.5 m
    85mm Whoop       8.5 m  5.4 m         18 m   10 m
    2.5" Freestyle   9.8 m  7.4 m         20 m   14 m
    5" Freestyle     11 m   10 m          22 m   20 m

  Tilted forward and holding height, a quad settles at a lower speed for
  the same tilt - the 65mm Whoop at 45 degrees cruises at 28 km/h instead
  of 31 - so you tilt a little further for the same speed (with the climb
  model below, 30). Flying forward it also needs a touch less throttle to
  hold height: tilted, the props' drag lies in their plane, so it points
  back and slightly up. Wind pushes a whoop around more, since its props
  drag on the wind as well.


CLIMBING
  Climbing, air comes through the props along their axis, faster the faster
  the quad climbs - and flying fast, tilted over, part of the air goes
  through them that way too. The game takes thrust off for that in a
  straight line from the first metre a second: at a third of the drone's
  top_speed a sixth of it is gone. And it counts the spinning props as more
  body drag along their axis as well (its corner drag: a full-throttle
  climb meets about 1.6 times the drag of a hover), on top of the thrust
  they have already lost. So a punch-out pulls hard for a quarter of a
  second and then fades.

  A real prop barely loses thrust at first - standing still its blades are
  partly stalled, and as the air comes they unload and the motor speeds up
  - and loses it faster nearer its pitch speed. With climb=true (the
  default) the thrust falls as 1 - 0.5 x (airspeed / top_speed)^1.5: still
  half at the drone's top_speed, as the game has it, so each drone's
  top_speed keeps its meaning, but with 90% left at a third of it rather
  than 83%, and 82% at half rather than 75%. That curve is fitted to a
  Gemfan 1811's bench test (45 mm, 1.8" pitch, on a 1102 motor - a Meteor75
  Pro II's props) with a prop's thrust against airspeed and the motor
  speeding up as the prop unloads: within 3% of it from 10 to 25 m/s. The
  props' extra drag along the axis is taken off while the air comes through
  them from above, since the thrust loss already counts what they cost a
  climb. Falling through them is left as the game has it (dirty_air works
  on that).

  Full throttle from a hover, after 1 second (the game's own quad script,
  measured in Godot; the 5" in the same test on the game's flight model):

    quad             height gained         speed reached
                     game   climb model    game   climb model
    65mm Whoop       8.5 m  9.7 m          43     52 km/h
    5" Freestyle     18.5 m 21.7 m         106    131 km/h

  Flat out at the best tilt and straight up, worked out with the game's
  flight model and rotor drag (km/h):

    quad             flat out               straight up
                     game   climb model     game   climb model
    65mm Whoop       50     60              46     58
    75mm Whoop       60     70              58     69
    85mm Whoop       67     80              62     78
    2.5" Freestyle   82     101             78     103
    3.5" Freestyle   107    125             104    126
    5" Freestyle     119    143             116    145
    5" Racer         136    168             134    171
    Beginner 5"      80     94              75     92

  A drone added by the drones mod has its own top_speed in its block: the
  airspeed through the props at which they have lost half their push. The
  Meteor75 Pro II's is 100 km/h, from its props, which flies it at about
  85 km/h flat out.


HOW IT WORKS OUT THE FORCE
  From momentum theory: props of total disc area A making thrust T move
  sqrt(1.225 x A x T / 2) kg of air a second (open), or sqrt(1.225 x A x T)
  in ducts. Part of the sideways momentum of that air is felt as drag:
    open props  0.17 of it - what a Crazyflie 2.0's measured rotor drag is
                (Forster 2015, system identification of the Crazyflie 2.0)
    ducts       0.25 of it - a quarter of the full ducted-fan figure, since
                a whoop's ducts are short and open at the bottom. This one
                is set by flying: against a real Meteor75 Pro II, the game's
                version slid further without rotor drag and stopped shorter
                at twice this
  The thrust is the game's own at that moment, so the drag follows the
  throttle (and battery_sag's sag); disarmed, the props are stopped and so
  is the drag.


SETTINGS
  Edit rotor_drag/settings.cfg (made at the first launch), save, restart the game.

  strength scales the rotor drag: 0 = none, 1 = the default, 2 = the 1.0
  release's (the 65mm Whoop above: half speed after 2.5 m), 4 = the whole
  ducted-fan momentum drag. If a whoop still slides further than a real
  one, try 1.3 or 1.5; if it feels heavy and stops short, 0.7.

  engine_damping is the engine's damping on the quad you fly: 0 = none (the
  default), 0.1 = as the game has it.

  climb=true (the default) is the climb model above; false = the game's own
  straight-line thrust loss and climb drag. strength=0, engine_damping=0.1
  and climb=false is the game exactly as it is.

  Each drone's prop size (prop_mm) and whether it has ducts are in
  drones/settings.cfg, which comes with this mod and is shared with
  dirty_air, motor_response and flight_controller, so each is set once. A
  drone added by the drones mod sets its own with prop_mm= and ducted= in its
  block (the Meteor75 Pro II: 46 mm, ducted).


MULTIPLAYER AND RACES
  Only the quad you fly is changed, on your computer; other players see
  where you fly as ever. Like any change to the flight model it changes how
  fast you get round a race track: the rotor drag makes a quad slower in
  the turns; the climb model makes it faster out of them and on the
  straights; without the engine's damping, climbs, falls and long coasts
  carry a little further.


INSTALL
  1. Steam: right-click The Zone FPV > Manage > Browse local files.
  2. Extract the zip into that folder, so override.cfg and the zonemods,
     drones and rotor_drag folders sit next to thezone.exe. Replace files
     when asked.

  rotor_drag is loaded by zonemods, which loads other mods alongside it - see
  zonemods/README.txt.


UNINSTALL
  To turn rotor_drag off without uninstalling it, set enabled=false at the top
  of rotor_drag/settings.cfg (or strength=0).

  Delete the rotor_drag folder. If no other mods use zonemods, delete override.cfg
  and the zonemods folder too.


TROUBLESHOOTING
  rotor_drag/status.txt is rewritten at every launch: the strength, the climb
  model, and a line each time you fly a different drone with its props and
  rotor drag. If a game update changes the game's table of thrust against
  airspeed, the climb model leaves it alone and status.txt says so.
  zonemods/status.txt lists which mods loaded.
