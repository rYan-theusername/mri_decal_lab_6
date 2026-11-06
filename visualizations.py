import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colormaps
from IPython.display import HTML


def plot_sequence(seq, time_range=None, title='Sequence diagram'):
    """ one row each for the RF pulse, the ADC, and the three gradients.
        time_range=(start, stop) in s zooms in; the default shows the whole sequence. """

    gx, gy, gz, rf = seq.waveforms_and_times(append_RF=True)[0]
    t_adc = seq.adc_times()[0]
    start, stop = time_range if time_range is not None else (0, seq.duration()[0])
    t_adc = t_adc[(t_adc >= start) & (t_adc <= stop)]

    fig, axes = plt.subplots(5, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(rf[0].real * 1e3, np.abs(rf[1]), color='tab:blue')
    axes[0].set_ylabel('RF\n(Hz)')
    axes[1].vlines(t_adc * 1e3, 0, 1, color='tab:red', linewidth=0.5)
    axes[1].set_ylabel('ADC')
    axes[1].set_yticks([])
    for ax, g, name in zip(axes[2:], (gx, gy, gz), 'xyz'):
        ax.plot(g[0].real * 1e3, g[1].real / 1e3, color='tab:green')
        ax.set_ylabel(f'$G_{name}$\n(kHz/m)')
    for ax in axes:
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.4)
    axes[0].set_xlim(start * 1e3, stop * 1e3)
    axes[0].set_title(title)
    axes[-1].set_xlabel('time (ms)')
    plt.tight_layout()
    plt.show()


def plot_signal(t, signals, names=None, TE=None, title='Received signal $|s(t)|$'):
    """ plot one or more |s(t)| traces against t (s). optional dashed line at TE. """

    if not isinstance(signals, list):
        signals = [signals]
    if names is None:
        names = [None] * len(signals)

    plt.figure(figsize=(8, 4))
    for s, name in zip(signals, names):
        plt.plot(t * 1e3, np.abs(s), linewidth=2, label=name)
    if TE is not None:
        plt.axvline(TE * 1e3, color='gray', linestyle='--', label=rf'$T_E$ = {TE*1e3:.0f} ms')
    plt.xlabel('time since the center of the $90^\\circ$ (ms)')
    plt.ylabel(r'$|s(t)|$')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    if names[0] is not None or TE is not None:
        plt.legend()
    plt.show()


def animate_transverse_plane(mxy, t, fps=10, every=1, title='Transverse plane'):
    """ arrows in the xy plane, one per spin, plus their sum in black.
        mxy is (n_spins, n_samples) complex, t is (n_samples,) in s. """

    n_spins, n_samples = mxy.shape
    s = mxy.sum(axis=0)
    colors = colormaps['coolwarm'](np.linspace(0, 1, n_spins))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('$M_x$')
    ax.set_ylabel('$M_y$')
    ax.grid(True, linestyle='--', alpha=0.4)

    def arrow(z, color, width):
        return ax.quiver(0, 0, z.real, z.imag, angles='xy', scale_units='xy',
                         scale=1, width=width, color=color)

    spins = [arrow(mxy[j, 0], colors[j], 0.008) for j in range(n_spins)]
    net = arrow(s[0] / n_spins, 'black', 0.014)  # the average, so it fits on the axes
    label = ax.set_title(title)

    def update(frame):
        i = frame * every
        for j, q in enumerate(spins):
            q.set_UVC(mxy[j, i].real, mxy[j, i].imag)
        net.set_UVC(s[i].real / n_spins, s[i].imag / n_spins)
        label.set_text(rf'{title}:  $t$ = {t[i]*1e3:.1f} ms,  $|s|$ = {abs(s[i]):.1f}')
        return spins + [net, label]

    anim = FuncAnimation(fig, update, frames=n_samples // every, interval=1000 / fps)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_kspace_samples(ktraj, title='where the ADC sampled k-space'):
    """ scatter the (kx, ky) of every ADC sample. ktraj is (3, n_samples) in 1/m. """

    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(ktraj[0], ktraj[1], s=2)
    plt.gca().set_aspect('equal')
    plt.xlabel('$k_x$ (1/m)')
    plt.ylabel('$k_y$ (1/m)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()


def show_images(images, titles, figsize=None):
    """ magnitude images side by side, in grayscale, with titles. """

    fig, axes = plt.subplots(1, len(images), figsize=figsize or (4 * len(images), 4))
    for ax, img, title in zip(np.atleast_1d(axes), images, titles):
        ax.imshow(np.abs(img), cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
