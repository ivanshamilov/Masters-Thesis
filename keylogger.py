from pynput import keyboard
import time


def on_key_release(key): #what to do on key-release
  global CURR_TIME, WAS_RELEASED, PREV_KEY
  CURR_TIME = round(time.time() - CURR_TIME, 10)
  print(f"The key {key} was pressed for {CURR_TIME}")
  WAS_RELEASED = True

def on_key_press(key): #what to do on key-press
  global CURR_TIME, WAS_RELEASED, PREV_KEY

  if WAS_RELEASED:
    CURR_TIME = time.time()
    print(f"{key} is pressed")
    WAS_RELEASED = False
    PREV_KEY = key
  else:
    return

if __name__ == "__main__":
  WAS_RELEASED = True
  PREV_KEY = keyboard.KeyCode("")
  CURR_TIME = time.time()

  with keyboard.Listener(on_press = on_key_press, on_release = on_key_release) as listener: #setting code for listening key-press
      listener.join()