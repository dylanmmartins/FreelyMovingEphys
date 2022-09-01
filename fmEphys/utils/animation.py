"""Animation using parallel processing.
"""

import subprocess
import wave

import ray
import numpy as np
from ray.actor import ActorHandle
from asyncio import Event
from typing import Tuple
from time import sleep
from asyncio import Event
from typing import Tuple
from time import sleep

@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter

class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return

def fig_to_img(fig):
    width, height = fig.get_size_inches() * fig.get_dpi()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(),
                    dtype='uint8').reshape(int(height), int(width), 3)

    return img

def stack_animation(pb, result_ids, savepath, speed=1):
    """
    `speed` will animate the video slower or faster than real time. positive values make it slower
        e.g. speed=1 is real-time
             speed=0.25 is quarter speed
             speed=2 is twice real-time
    """

    # Progressbar and get results
    pb.print_until_done()
    results_p = ray.get(result_ids)
    images = np.stack([results_p[i] for i in range(len(results_p))])

    # Make video with opencv
    fps = slow_by * np.size(images,0) * 1000
    out = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (images.shape[-2], images.shape[-3]))

    for f in range(np.size(images,0)):
        out.write(cv2.cvtColor(images[f], cv2.COLOR_BGR2RGB))
    out.release()

def note_to_freq(n):
    """
    Map index of piano key to frequency
    """
    freq = 2 ** ((n-49)/12)
    return freq * 440 # Hz

def make_solo_tone(eventT, tlen, tone=40, tone_lasts=30):
    """
    tlen: sec of audio to write
    tone_lasts: how many msec long should each tone be?
    """
    rate = 30000
    t = np.linspace(0, tlen, int(tlen*rate), endpoint=True)
    tone_len = int(rate*(tone_lasts/1000))
    tone_t = np.arange(0, tone_len)

    freq = note_to_freq(tone)

    tone = np.sin(2*np.pi * freq * tone_t)

    sound_arr = np.zeros(np.size(t))

    for s in eventT:
        t_on = int(s*rate)
        t_off = int((s*rate)+tone_len)
        if t_off < np.size(x):
            sound_arr[t_on:t_off] = tone

    return sound_arr

def make_spike_sounds(spikeT, tlen=3):
    """
    spikeT:
    tlen: seconds of time that spikes range (i.e. a 3 sec clip of spike times)
    """
    rate = 30000
    t = np.linspace(0, tlen, int(tlen*rate), endpoint=True)

    # play each spike sound for 30 ms
    sp_len = int(rate*(30/1000)) 
    tone_t = np.arange(0, sp_len)

    sinlist = []

    for count, sps in enumerate(spikeT.values()):
        
        freq = note_to_freq(((count+1)*2))
        
        tone = np.sin(2*np.pi * freq * tone_t)
        
        x = np.zeros(np.size(t))
        
        for sp in sps:
            t_on = int(sp*rate)
            t_off = int((sp*rate)+sp_len)
            if t_off < np.size(x):
                x[t_on:t_off] = tone
        
        sinlist.append(x)

    sound_arr = np.zeros(int(rate*tlen))

    for s in sinlist:
        sound_arr = np.add(sound_arr, s)

    return sound_arr

def write_wav(sound_arr, savepath, tlen, nSamps, rate=30000, speed=1):
    """
    rate = 0.25 is quarter speed
    """
    
    wav_file = wave.open(fname, "w")

    nchannels = 1
    sampwidth = 2
    comptype = "NONE"
    compname = "not compressed"

    wav_file.setparams((nchannels, sampwidth, int(rate*4), nSamps, comptype, compname))
    
    sound_arr = sound_arr / np.max(sound_arr)
    # increase so it ranges from min:max of int16 dtype
    sound_arr = sound_arr*(np.iinfo(np.int16).max)

    # write the audio frames to file
    wav_file.writeframes(sound_arr.astype(np.int16))
    # int(s*amp/2)))

    wav_file.close()

def trim_mp3(path, savepath, startT, win_len):
    """
    startT and stopT should be time in sec
    """
    subprocess.call(['ffmpeg', '-ss', str(startT), '-t', str(win_len),
                            '-i', path, '-acodec', 'copy', savepath])

def merge_video_audio(video_path, audio_path, savepath):
    subprocess.call(['ffmpeg', '-i', video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-y', savepath])
