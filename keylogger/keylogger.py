import csv
import json
import sys
import threading
import time
from contextlib import contextmanager
from typing import List, Optional, Union, Dict

import pynput
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from special_symbols import PYNPUT_TO_DEFAULT


class InputController(object):
  @contextmanager
  def capture_input(self) -> List:
    _data = []

    def reader():
      while reader.running:
        _data.append(sys.stdin.readline()[:-1])
    reader.running = True
    thread = threading.Thread(target=reader)
    thread.start()
    try:
      yield _data
    finally:
      reader.running = False
      thread.join()


class KeyLogger(object):
  def __init__(self, mapping: Dict[str, int], initial_sentence: str) -> None:
    self.initial_sentence = initial_sentence
    self.presses = list()
    self.releases = list()
    self.curr = 0
    self.counter = {}
    self.aligner = list()
    self._columns = ["SENTENCE", "USER_INPUT", "KEYCODE", "KEY", "PRESS_TIME", "RELEASE_TIME"]
    self.input_controller = InputController()
    self.collected_input = ""
    self.mapping = mapping

  def _parse_key(self, key: Union[KeyCode, Key]) -> str:
    _key = None
    match type(key):
      case pynput.keyboard._darwin.KeyCode: _key = key.char.lower()
      case _: _key = key.name # add numpads, virtual keys range from 96 to 105 (96 - 1, ..., 105 - 9)
    return PYNPUT_TO_DEFAULT.get(_key, _key)
    
  def on_key_release(self, key: Union[KeyCode, Key]) -> None:
    # print(f"The key {key} was released at {time.strftime('%b %d %Y %H:%M:%S', time.gmtime(time.time()))}")
    key = self._parse_key(key)
    self.aligner.append(self.counter[key])
    self.releases.append([key, time.time()])

  def on_key_press(self, key: Union[KeyCode, Key]) -> None:
    if key == Key.enter:
      print(self.aligner)
      return False
    # print(f"The key {key} was pressed at {time.strftime('%b %d %Y %H:%M:%S', time.gmtime(time.time()))}")
    key = self._parse_key(key)
    self.presses.append([key, time.time()])
    self.curr += 1
    self.counter[key] = self.curr

  def _align_releases(self) -> None:
    _, self.releases = zip(*sorted(zip(self.aligner, self.releases)))
    self.releases = list(self.releases)

  def concat_results(self) -> None:
    self.result = [[code, *x, y[1]] for (x, y, code) in zip(self.presses, self.releases, 
                                                      list(map(lambda x: self.mapping[x[0]], self.presses))
                                                      )]

  def listen(self) -> None:
    with self.input_controller.capture_input() as collected_input:
      with keyboard.Listener(on_press=self.on_key_press,
                             on_release=self.on_key_release) as listener:
        print(f"Sentence to write: {self.initial_sentence}")
        listener.join()
    self.collected_input = collected_input[0]
    self._align_releases()
    self.concat_results()

  def to_csv(self, filepath: str) -> None:
    assert hasattr(self, "result"), "Nothing has been typed yet"
    with open(filepath, "w") as f:
      writer = csv.writer(f)
      writer.writerow(self._columns)
      for row in self.result:
        row.insert(0, self.initial_sentence)
        row.insert(1, self.collected_input)
        writer.writerow(row)


if __name__ == "__main__":
  MAPPING = dict()
  with open("mappings/key-codes.json", "rb") as f:
      MAPPING = json.load(f)

  keylogger = KeyLogger(mapping=MAPPING, initial_sentence="Hello World! I am Ivan :)")
  keylogger.listen()
  keylogger.to_csv("hello.csv")
