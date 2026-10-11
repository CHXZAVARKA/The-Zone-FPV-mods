player_list 1.0
=================

Keeps the game's list of the other players - their names, and the buttons to
watch them in a small window or full screen - out of the picture, and says in
the chat who joins and leaves the lobby.


THE LIST (player_list)
  "flyout" (as shipped) - out of the way until you move the mouse to the left
  edge of the screen, in the top two thirds. Then it slides out on a dark
  panel, and you can click its buttons as usual; it slides back once the
  mouse moves off it, or when the game hides the pointer (3 s without the
  mouse moving). With nobody else in the lobby it says so.

  "hidden" - never shown.

  "shown" - as the game shows it.


JOIN AND LEAVE MESSAGES (join_leave_messages)
  On as shipped. In the chat, in gold so they read apart from what players
  say:
    In this lobby: Alice, Bob and Carol     (as you arrive)
    Dave joined
    Alice left
    Alice rejoined                          (back within two minutes)
  Players who come or go within a second of each other share a message.
  More than three at once, and it gives how many instead of who: "24 other
  players in this lobby", "5 players left".
  The game lets a player go 5 seconds after it last hears from them, so
  "left" comes that long after they go - and if your own connection drops,
  everyone "left" and then "rejoined".

Both change at once in Pause > Mods with mod_settings, or in
player_list/settings.cfg (then restart the game). player_list/status.txt says what
was applied.


INSTALL
  1. Steam: right-click The Zone FPV > Manage > Browse local files.
  2. Extract the zip into that folder, so override.cfg and the zonemods and
     player_list folders sit next to thezone.exe. Replace files when asked - your
     settings are kept.

  player_list is loaded by zonemods, which loads other mods alongside it - see
  zonemods/README.txt.


UNINSTALL
  Set enabled=false at the top of player_list/settings.cfg, or delete the player_list
  folder. If no other mods use zonemods, delete override.cfg and the zonemods
  folder too.
