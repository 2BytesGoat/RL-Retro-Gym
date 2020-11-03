import time
import retro
import pandas as pd
from pynput import keyboard
from gym.wrappers import Monitor

class ManualController():
    def __init__(self, env, rec_path='./video'):
        self.rec_path = rec_path
        self.env = Monitor(env, rec_path, force=True, 
                    video_callable=lambda episode_id: True)
        self._delay_ms = 15 # magic number for displaying the game real-time
        self.record = True
        self.episode_summary = None
        self.action = [0] * len(env.buttons)
        self.key_mapping = self._get_key_mapping(env.buttons)

    def _get_key_mapping(self, buttons):
        key_mapping = {
            'w': buttons.index('UP'),
            's': buttons.index('DOWN'),
            'a': buttons.index('LEFT'),
            'd': buttons.index('RIGHT'),
            'n': buttons.index('A'),
            'm': buttons.index('B'),
            'j': buttons.index('X'),
            'k': buttons.index('Y')
        }
        return key_mapping

    def _set_key(self, key):
        if key in self.key_mapping.keys():
            key_index = self.key_mapping[key]
            self.action[key_index] = 1
            
    def _unset_key(self, key):
        if key in self.key_mapping.keys():
            key_index = self.key_mapping[key]
            self.action[key_index] = 0

    def _store_summary(self, info, action):
        tmp_info = info.copy()
        tmp_info['action'] = str(action)
        if self.episode_summary is None:
            self.episode_summary = pd.DataFrame(tmp_info, index=[0])
        else:
            info_df = pd.DataFrame(tmp_info, index=[len(self.episode_summary)])
            self.episode_summary = pd.concat([self.episode_summary, info_df])

    def _save_summary(self, episode_nb=0):
        self.episode_summary.to_csv(f'{self.rec_path}/openaigym.episode_summary.{episode_nb}.csv')

    def on_press(self, key):
        if key == keyboard.Key.esc:
            # Stop listener
            self.record = False
            return False
        try:
            self._set_key(key.char)
        except AttributeError:
            # Special char go brrr
            pass

    def on_release(self, key):
        try:
            self._unset_key(key.char)
        except AttributeError:
            # Special char go brrr
            pass

    def start_key_listener(self):
        # Collect events until released
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.start()

    def play(self):
        self.start_key_listener()
        for episode_nb in range(5):
            if not self.record:
                # If the ESC button is pressed
                # before the last episode
                break
            done = False
            self.env.reset()
            # Run one episode until its done
            while self.record and not done:
                self.env.render()
                _, _, done, info = self.env.step(self.action)
                self._store_summary(info, self.action)
                time.sleep(self._delay_ms/1000)
            self._save_summary(episode_nb)
        self.env.close()

if __name__ == '__main__':
    env = retro.make(game='SuperMarioKart-Snes')
    mc_hammer = ManualController(env, rec_path='./video')
    mc_hammer.play()