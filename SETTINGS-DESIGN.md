# Settings: where they live, and what to do about it

Notes from going through all 13 mods' settings in October 2026, after a week
of chasing a prop wash problem that turned out to be three settings in three
mods. Written down so the next person - or the next session - does not have
to re-derive it.

## The shape of the problem

357 settings across 13 mods. To change how the quad flies you visit seven of
them: `dirty_air`, `flight_controller`, `drones`, `rotor_drag`, `battery_sag`,
`prop_damage`, `mode_3d`. The settings UI is organised by mod because the
files are organised by mod. Nobody thinks "which mod owns this" - they think
"the air feels wrong".

Of those 357, the ones that actually changed the feel during that week were
about ten: `dirty_air`'s `descent`, `wake`, `wash`, `turbulence` and `gusts`,
`flight_controller`'s `enabled`, `gyro_noise` and `airmode`, and the per-drone
figures in `drones`.

## Three layers, and where they are mixed up

The suite half-follows a clean split already:

| layer | lives in | examples |
| --- | --- | --- |
| **facts** - what the quad *is* | `drones` | `prop_mm`, `ducted`, `wheelbase_mm`, `motor_ms` |
| **laws** - what those facts mean | each physics mod, in code | `VRS_WIDTH`, `AMBIENT_AIR`, `WAKE_SPREAD` |
| **taste** - what the player wants | each mod's own settings | `turbulence`, `wash`, `shake` |

Two places break it:

- `wash` and `wake` exist **twice**: once in `dirty_air` as taste, and once
  per drone in `drones` as a multiplier. The per-drone copies are taste
  sitting in the facts table. They should be overrides for when a law is
  wrong about one build, named and documented as such - not a second place to
  tune feel.
- `gyro_noise` lives in `flight_controller`, but how much motor vibration
  gets through a frame is a **fact about the airframe**, like `motor_ms`. It
  belongs in `drones`, read by `flight_controller`, exactly as `motor_ms`
  already is. (A real Meteor75 measures about 4; a clean 5 inch would differ.)

Moving a setting between mods orphans whatever players already have in their
`settings.cfg`, so either move needs a migration path. `zonemods` keeps a
value only when it differs from `settings.base.cfg`, which handles *changed
defaults* but not *moved keys*.

## Grouping the UI by domain instead of by mod

`cfgdoc.gd` already groups settings *within* a page: a `;; ---- Title ----`
line starts a group. What is missing is a domain above the mod, so one page
can draw from several mods.

Suggested domains, with what feeds them:

- **Air** - `dirty_air`, `rotor_drag`
- **Flight controller** - `flight_controller`
- **The drone** - `drones`, `battery_sag`
- **Wear** - `prop_damage`
- **Modes** - `mode_3d`, `throttle_fix`
- **Picture and sound** - `fpv_osd`, `analog400mw`, `pilot_audio`
- **Interface** - `player_list`, `mod_settings`

Smallest version that would have paid for itself: a **flight feel** page with
the ten settings above and everything else behind "more".

## Acceptable values

`cfgdoc.gd` reads a range out of the note already. `_range()` takes the first
`N to M` in the lines directly above a key and accepts it when `lo < hi` and
the default falls inside, so writing `0 to 3` in the prose is all it takes.

Only **2 of the 220** numeric settings in the suite said a range, so the form
had nothing to go on for the other 218. `dirty_air`'s eight now do.

The rest are better written by whoever knows what each setting means - a
wrong range is worse than none, because it reads as advice. The code's own
clamp stays a backstop: reading the range from the note inside each mod would
make every mod depend on `cfgdoc.gd`, which belongs to `mod_settings` and is
not there when `mod_settings` is not installed.

Watch out for an accidental `N to M` earlier in a note (`"about 2 to 3
times"`), since the first match wins.

## Restart flags

Only `drones` declares any. Everything else comes from two blanket rules:

1. `mod_settings/main.gd` `is_live()` returns false for `mod/enabled`, always.
2. A mod with no `zm_apply_settings` has all of its settings need a restart.
   Only `throttle_fix` lacks one, and it has no settings besides `enabled`,
   so this costs nothing today.

Rule 1 is **not** conservatism. `zonemods/zonemods.gd` returns before loading
a mod's script when `enabled=false`, so a disabled mod has no node - there is
nothing for a setting to be applied to. Making `enabled` live needs all of:

- `zonemods` loading every mod whatever its `enabled` says;
- each of the 13 mods doing nothing, and undoing whatever it had done, when
  it is off - different work in each, and `fpv_osd` (UI), `dirty_air`
  (forces) and `flight_controller` (a mounted script) are three different
  problems;
- the four mods that mount a resource pack - `drones`,
  `flight_controller`, `analog400mw`, `mode_3d` - mounting it always and
  passing through when off, because `load_resource_pack` and the `.remap`
  substitution have to happen before the game loads its own scripts. That
  means the pack is mounted for players who have the mod off, so a game
  update that breaks a mount would reach everyone rather than only the people
  who opted in. The code already falls back cleanly when the game's scripts
  are not what it expects, which is what makes this tolerable rather than
  reckless.

What is left after that is **one** restart flag: adding or removing a drone
in `my_drones.cfg`, because the game builds its menus from the vehicle list
at startup.

So it is worth doing, and it is a refactor across every mod rather than a
rule to delete.

## Order worth doing it in

Each of these changes how the game behaves at startup or how the settings
page draws, and neither can be checked without running the game. One at a
time, with a flight in between:

1. **Per-setting ranges** for the remaining mods. No behaviour change, and
   the UI improves as each one lands.
2. **The flight feel page**. Pure addition to `mod_settings`; nothing else
   moves.
3. **Domain grouping**, once the page above proves the rendering.
4. **`gyro_noise` into `drones`**, as the first and smallest layer fix, with
   a migration for existing `settings.cfg`.
5. **`wash`/`wake` per-drone copies** renamed and documented as overrides.
6. **Live `enabled`**, last, because it touches `zonemods` and all 13 mods.
   If a mount regresses here, nothing loads.
