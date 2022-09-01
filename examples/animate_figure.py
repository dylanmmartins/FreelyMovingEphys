"""Example figure animation.

Animate a sequence of plots/images into a video. The plot/image itself is made
with matplotlib, and the animated .mp4 video is written with OpenCV. This uses
the Ray for parallel processing so that the sequence of images can be plotted
in parallel, and then stacked together into a video once all images are ready.

Some functions are read in from

The basic steps are to


I haven't defined `t_vals`, `x`, `y1`, or `y2` in this example. The array `t_vals`
would be a sequence of indexes or timestamps for each frame, and the x and y variables
in this example animation are arrays of values that the function `plot_frame_img()`
would visualize in a matplotlib plot.

"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import ray
from ray.actor import ActorHandle

import fmEphys

@ray.remote
def plot_frame_img(t, x, y1, y2, pbar:ActorHandle,):

    ### Set up the figure
    fig, [ax0,ax1] = plt.subplots(1,2, constrained_layout=True, figsize=(3,3), dpi=200)
    
    ### Plot variables
    ax0.plot(x, y1[t,:,:])
    ax1.plot(x, y2[t,:,:])

    ### Save the figure out as an image
    plt.tight_layout()

    img = fmEphys.utils.animation.fig_to_img(fig)
    
    plt.close()
    pbar.update.remote(1)
    return img

mpl.use('agg')

numFr = len(t_vals)

pb = fmEphys.utils.animation.ProgressBar(numFr)
actor = pb.actor

x_r = ray.put(x)
y1_r = ray.put(y1)
y2_r = ray.put(y2)

result_ids = []

for t in t_vals:
    # Append the parameters for each frame into a list, `result_ids`
    result_ids.append(plot_frame_img.remote(t, x_r, y1_r, y2_r, actor))

# Using the list of parameters for each frame, we'll animate all of the frames,
# create an image for each, stack the images, and then save it out as an mp4 file.
fmEphys.utils.animation.stack_animation(pb, result_ids, '/path/to/save/file.mp4', speed=0.25)