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
    if type(key) == Key:
      _key = key.name
    else:
      _key = key.char.lower()
    return PYNPUT_TO_DEFAULT.get(_key, _key)
    
  def on_key_release(self, key: Union[KeyCode, Key]) -> None:
    # print(f"The key {key} was released at {time.strftime('%b %d %Y %H:%M:%S', time.gmtime(time.time()))}")
    key = self._parse_key(key)
    self.aligner.append(self.counter[key])
    self.releases.append([key, time.time() * 10**3])

  def on_key_press(self, key: Union[KeyCode, Key]) -> None:
    if key == Key.enter:
      return False
    # print(f"The key {key} was pressed at {time.strftime('%b %d %Y %H:%M:%S', time.gmtime(time.time()))}")
    key = self._parse_key(key)
    self.presses.append([key, time.time() * 10**3])
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

  # sentences = [
  #   "Few black taxis drive up major roads on quiet hazy nights.",
  #   "Levi Lentz packed my bag with six quarts of juice.",
  #   "A quick brown fox jumps over a lazy dog.",
  #   "Bobby Klum awarded Jayme sixth place for her very high quiz.",
  #   "Back in June we delivered oxygen equipment of the same size.",
  #   "J.Fox made five quick plays to win the big prize.",
  #   "I loved this movie the first time I saw it and on each subsequent viewing I always notice at least one new detail.",
  #   "Wasn't the shop located like right on the beach or something?",
  #   "It was worth all the money I gave for it!",
  #   "Am I the only one who sees this?",
  #   "I can't believe there are people out there that did not like this movie!",
  #   "This would have to be by far the greatest series I have ever seen."
  # ]

  # sentences = [
  #   "She hoped she wasn't about to get fired.",
  #   "The computer is dying on me.",
  #   "I don't like close spaces.",
  #   "This is my younger brother.",
  #   "The app wasn't loading for me.",
  #   "I have a name and a surname.",
  #   "I put on my underclothes, shirt and trousers.",
  #   "We won't be taking you with us.",
  #   "Does she drink coffee?",
  #   "We are starting to enter a new chapter.",
  # ]

  # sentences = [
  #   "The tables were made of fake wood.",
  #   "The bleeding isn't Tom's biggest problem.",
  #   "Hairless cats look demonic.",
  #   "My water bottle is white and made of steel.",
  #   "I'm going to crash at your place.",
  #   "The sun comes up in the east.",
  #   "He is only about six feet tall.",
  #   "He didn't get the position.",
  #   "Do you mind if I smoke?",
  #   "Good location choice.",
  # ]

  # sentences = [
  #   "The steak is on the grill.",
  #   "The children are at home.",
  #   "She had nothing else to say to him.",
  #   "I have a small house.",
  #   "He has made a big improvement in tennis.",
  #   "The baby was in a parka in the stroller.",
  #   "The pencil broke in the middle of my test.",
  #   "I am a football fan.",
  #   "He goes to school.",
  #   "Sit down and cross your legs, please!",
  # ]


  index = 8
  print("Start typing...")
  keylogger = KeyLogger(mapping=MAPPING, initial_sentence=sentences[index])
  keylogger.listen()
  keylogger.to_csv(f"{index+1}.csv")
