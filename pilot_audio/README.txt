pilot_audio 1.3
=================

Hear the quad from where you stand. The Zone FPV plays the motor sound as if
you were on board - as loud 300 m away as on the pad. pilot_audio puts you at the
spawn point, at head height, facing the way the spawn faces, and works out
what reaches you there - how loud, how muffled, how late and from which side:

  loudness   6 dB quieter each time the distance doubles
  air        duller the further away, as the air soaks up the highs
  walls      quieter and duller with a building or hill in the way
  delay      sound takes a moment to arrive, and the pitch rises as the quad
             comes towards you and drops as it goes away (doppler)
  stereo     left or right by where the quad is, a little duller behind you

In multiplayer, other players' quads get a motor sound too (the game has none
for them), heard the same way.

Or listen from the drone (listen_from="drone"): you ride along at your quad's
camera, turning as it turns. Your quad sounds as the game plays it, and other
players' quads are heard from where you fly - nearer and louder as you close
in, left or right of your camera, their pitch rising as you fly at each other.

In prop wash the motor sound warbles - its pitch and loudness jitter as the
motors hunt in the dirty air, more the rougher it is (propwash_sound; needs
dirty_air, which works out how rough the air is). Other players' quads
warble too when they fall through their own wake (propwash_others): pilot_audio
works it out from their attitude, throttle and how they move, as dirty_air
does for yours, so this part does not need dirty_air.

With prop_damage, a damaged prop - out of balance - makes the motor sound
rough: a fast flutter of pitch and loudness, more the worse the prop and the
harder it spins.

The spawn point is also where the game's analog video link is measured from,
so the picture and the sound agree. The game's S key moves the spawn to where
the quad is (land first - it keeps the quad's heading); X puts it back.


INSTALL
  1. Steam: right-click The Zone FPV > Manage > Browse local files.
  2. Extract the zip into that folder, so override.cfg and the zonemods and
     pilot_audio folders sit next to thezone.exe. Replace files when asked - your
     settings are kept (see UPDATING in zonemods/README.txt).

  pilot_audio is loaded by zonemods, which loads other mods alongside it - see
  zonemods/README.txt.


UNINSTALL
  To turn pilot_audio off without uninstalling it, set enabled=false at the top
  of pilot_audio/settings.cfg.

  Delete the pilot_audio folder. If no other mods use zonemods, delete override.cfg
  and the zonemods folder too.


SETTINGS
  Edit pilot_audio/settings.cfg (made at the first launch), save, restart the game.
  Each effect can be turned off, and how fast the sound fades, how wide the
  stereo is, which way you face, whether you listen from the spawn or the
  drone and how much prop wash is heard can be set. The game's volume slider
  still sets the overall level. With mod_settings, Pause > Mods > pilot_audio
  changes them in the game, at once.


OTHER PLAYERS
  The game sends each player's position and sticks, not their motor speed or
  whether they are armed. Their motor sound follows their throttle the way
  the game's does for your quad, and a quad sat still at zero throttle goes
  quiet. The nearest 8 are heard (multiplayer_max); as players come and go
  or fly nearer and further, sounds fade in and out rather than starting and
  stopping dead.


TROUBLESHOOTING
  pilot_audio/status.txt is rewritten at every launch and says what was set up;
  zonemods/status.txt lists which mods loaded.
