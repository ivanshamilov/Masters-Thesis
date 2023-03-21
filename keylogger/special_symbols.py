# SPECIAL_SYMBOLS = {  # align <shift> + <key> symbols to the actual keys
#   "!": "1",
#   "@": "2",
#   "#": "3",
#   "$": "4",
#   "%": "5",
#   "^": "6",
#   "&": "7",
#   "*": "8",
#   "(": "9",
#   ")": "0",
#   "_": "-",
#   "+": "=",
#   "{": "[",
#   "}": "]",
#   "|": "\\",
#   ":": ";",
#   "\"": "\'",
#   "<": ",",
#   ">": ".",
#   "?": "/"
# }

"""
pynput has its own representation of symbols (both keynames and keycode)
Thus, mapping between pynput keys and default keycodes is needed
"""

PYNPUT_TO_DEFAULT = {   # align key names used in pynput to the ones defined in the dataset
  # MODIFIER KEYS
  "shift_r": "shift",
  "alt_r": "alt",
  "caps_lock": "capslock",
  "shift_l": "shift",

  # ARROWS
  "left": "leftarrow",
  "right": "rightarrow",
  "down": "downarrow",
  "up": "uparrow",

  # NUMPADS
  
  # WINDOW KEYS
  # left window / right window key 

  # PUNCTUATION
  ",": "comma",
  ";": "semicolon",
  "\'": "singlequote",
  "\\": "backslash",
  "`": "graveaccent",
  "/": "forwardslash",
  ".": "period",
  "-": "dash",
  "=": "equalsign",
  "[": "openbracket",
  "]": "closebracket",
  # <shift> + key
  "!": "1",
  "@": "2",
  "#": "3",
  "$": "4",
  "%": "5",
  "^": "6",
  "&": "7",
  "*": "8",
  "(": "9",
  ")": "0",
  "_": "dash",
  "+": "equalsign",
  "{": "openbracket",
  "}": "closebracket",
  "|": "backslash",
  ":": "semicolon",
  "\"": "singlequote",
  "<": "comma",
  ">": "period",
  "?": "forwardslash"
}
