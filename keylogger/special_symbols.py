from pynput.keyboard import KeyCode

SPECIAL_SYMBOLS = {
  "!": KeyCode.from_char("1"),
  "@": KeyCode.from_char("2"),
  "#": KeyCode.from_char("3"),
  "$": KeyCode.from_char("4"),
  "%": KeyCode.from_char("5"),
  "^": KeyCode.from_char("6"),
  "&": KeyCode.from_char("7"),
  "*": KeyCode.from_char("8"),
  "(": KeyCode.from_char("9"),
  ")": KeyCode.from_char("0"),
  "_": KeyCode.from_char("-"),
  "+": KeyCode.from_char("="),
  "{": KeyCode.from_char("["),
  "}": KeyCode.from_char("]"),
  "|": KeyCode.from_char("\\"),
  ":": KeyCode.from_char(";"),
  "\"": KeyCode.from_char("\'"),
  "<": KeyCode.from_char(","),
  ">": KeyCode.from_char("."),
  "?": KeyCode.from_char("/")
}