import sys

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

mode = sys.argv[1]
idx = int(sys.argv[2])

width = 8
height = 8
channel = 3
num_camera = 4

plot_len = 10
scale_position = 10
fps = 10

f_h5 = h5py.File('data.h5', 'r')

motion = f_h5['motion'][mode][()][idx]
position = f_h5['position'][mode][()][idx] * scale_position
vision = f_h5['vision'][mode][()][idx]

vision = vision.reshape(-1, channel, width * num_camera, height)
vision = vision.transpose(0, 2, 3, 1)

seq_len = motion.shape[0]

fig = plt.figure(figsize=(12, 8))

gs = GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])

gs_p = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs[0:2, 0])
ax_p = fig.add_subplot(gs_p[:, :])
gs_m = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs[0:2, 1])
ax_m0 = fig.add_subplot(gs_m[0, :])
ax_m1 = fig.add_subplot(gs_m[1, :])

gs_v = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs[2, 1])
ax_v = fig.add_subplot(gs_v[:, :])

ax_m0.set_xlim(0, plot_len - 1)
ax_m1.set_xlim(0, plot_len - 1)
ax_m0.set_ylim(-1.1, 1.1)
ax_m1.set_ylim(-1.1, 1.1)

ax_p.set_xlim(-scale_position * 1.3, scale_position * 1.3)
ax_p.set_ylim(-scale_position * 1.3, scale_position * 1.3)

ax_v.set_title('vision')
ax_p.set_title('position')
ax_m0.set_title('motion')


def get_img(t):
    img = vision[t]
    img = numpy.concatenate(numpy.split(img, 4, 0), 1)
    return img


v = ax_v.imshow(get_img(0), animated=True)
m0, = ax_m0.plot(range(plot_len), motion[:plot_len, 0])
m1, = ax_m1.plot(range(plot_len), motion[:plot_len, 1])

ax_p.scatter(scale_position, scale_position, marker='s', color='r', s=500)
ax_p.scatter(scale_position, -scale_position, marker='s', color='y', s=500)
ax_p.scatter(-scale_position, -scale_position, marker='s', color='b', s=500)
ax_p.scatter(-scale_position, scale_position, marker='s', color='g', s=500)

p = ax_p.scatter(position[0, 0], position[0, 1], marker='s', color='k')
p_line, = ax_p.plot(position[0], color='k')


def update(t):
    v.set_array(get_img(t))
    p.set_offsets(position[t])
    plot_begin = t - plot_len if t > plot_len else 0
    plot_end = t + 1
    m0.set_data(range(plot_begin, plot_end), motion[plot_begin:plot_end, 0])
    m1.set_data(range(plot_begin, plot_end), motion[plot_begin:plot_end, 1])
    p_line.set_data(position[:t + 1, 0], position[:t + 1, 1])
    ax_m0.set_xlim(t - plot_len, t)
    ax_m1.set_xlim(t - plot_len, t)
    ax_m0.set_xticks(range(t - plot_len, t + 1))
    ax_m1.set_xticks(range(t - plot_len, t + 1))


ani = animation.FuncAnimation(
    fig, update, range(seq_len), interval=1000. / fps)

plt.show()
