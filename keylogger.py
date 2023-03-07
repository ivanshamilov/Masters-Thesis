from pynput import keyboard
from pynput.keyboard import Key
import time
from queue import Queue


class KeyLogger:
  def __init__(self, initial_sentence=""):
    self.initial_sentence = initial_sentence
    self.presses = list()
    self.releases = list()
    self.curr = 0
    self.counter = {}
    self.aligner = list()

    print(f"Sentence to write: {self.initial_sentence}")

  def on_key_release(self, key):
    print(f"The key {key} was released at {time.strftime('%b %d %Y %H:%M:%S', time.gmtime(time.time()))}")
    self.aligner.append(self.counter[key])
    self.releases.append((key, time.time()))

  def on_key_press(self, key):
    if key == Key.enter:
      return False
    print(f"The key {key} was pressed at {time.strftime('%b %d %Y %H:%M:%S', time.gmtime(time.time()))}")
    self.presses.append((key, time.time()))
    self.curr += 1
    self.counter[key] = self.curr

  def _align_releases(self):
    _, self.releases = zip(*sorted(zip(self.aligner, self.releases)))
    self.releases = list(self.releases)

  def listen(self):
    with keyboard.Listener(on_press = self.on_key_press, on_release = self.on_key_release) as listener: #setting code for listening key-press
      listener.join()
    self._align_releases()
  
  
if __name__ == "__main__":
  keylogger = KeyLogger(initial_sentence="hello world")
  keylogger.listen()

  print(keylogger.presses)
  print(keylogger.releases)
