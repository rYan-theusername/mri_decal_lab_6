import os
import tempfile
from contextlib import contextmanager

import numpy as np
import pypulseq as pp
import MRzeroCore as mr0

# MRZero downloads its brain phantom and writes a temporary .seq file.
# we keep all of that out of the lab folder.
CACHE = os.path.join(tempfile.gettempdir(), 'mri_decal_mr0')
os.makedirs(CACHE, exist_ok=True)


def centered_ifft(s):
    """ centered IFFT from Lab 5. k = 0 and x = 0 land in the middle. """
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(s)))


@contextmanager
def _quietly_in_cache():
    """ run MRZero from CACHE, and hide the debug text its Rust core prints. """
    cwd, devnull, stdout = os.getcwd(), os.open(os.devnull, os.O_WRONLY), os.dup(1)
    os.chdir(CACHE)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(stdout, 1)
        os.close(devnull)
        os.close(stdout)
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def center_time(event):
    """ seconds from the start of an RF pulse's or ADC's block to its center. """
    if event.type == 'adc':
        return event.delay + event.num_samples * event.dwell / 2
    return event.delay + event.center


def delay_block(seconds):
    """ a delay block, rounded to the 10 us block raster the scanner requires. """
    return pp.make_delay(round(seconds / 1e-5) * 1e-5)


def adc_times(seq):
    """ time of every ADC sample, in seconds from the start of seq. """
    t0, times = 0.0, []
    for block_id in seq.block_events:
        block = seq.get_block(block_id)
        if block.adc is not None:
            adc = block.adc
            times.append(t0 + adc.delay + (np.arange(adc.num_samples) + 0.5) * adc.dwell)
        t0 += pp.calc_duration(block)
    return np.concatenate(times)


def make_gre(system, TE, TR, flip_deg, N=64, fov=256e-3, spoil=True, n_dummy=100):
    """ 2D Cartesian gradient echo, N by N, the same five blocks you wrote in Part 3.

        spoil=True adds, every TR:
            + RF spoiling: the RF (and receiver) phase changes as 117 * (i^2 + i + 2) degrees
            + gradient spoilers on x and z after the readout
            + a y rewinder that undoes this line's phase encode
        and plays n_dummy TRs with no ADC first, so the signal reaches steady state. """

    seq = pp.Sequence(system)
    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_deg * np.pi / 180, duration=2e-3, slice_thickness=5e-3, apodization=0.5,
        time_bw_product=4, system=system, use='excitation', return_gz=True,
        delay=system.rf_dead_time,
    )
    dk = 1 / fov
    gx = pp.make_trapezoid('x', flat_area=N * dk, flat_time=3.2e-3, system=system)
    adc = pp.make_adc(N, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid('x', area=-gx.area / 2, duration=1e-3, system=system)
    gx_spoil = pp.make_trapezoid('x', area=2 * N * dk, system=system)
    gz_spoil = pp.make_trapezoid('z', area=4 / 5e-3, system=system)
    phase_areas = (np.arange(N) - N // 2) * dk

    # same TE and TR bookkeeping as Part 3, plus the spoiler block
    delay_te = (TE - (pp.calc_duration(rf, gz) - center_time(rf))
                - pp.calc_duration(gx_pre, gz_reph) - center_time(adc))
    delay_tr = (TR - pp.calc_duration(rf, gz) - pp.calc_duration(gx_pre, gz_reph)
                - delay_te - pp.calc_duration(gx))
    if spoil:
        delay_tr -= pp.calc_duration(gx_spoil, gz_spoil)

    first = -n_dummy if spoil else 0
    for i in range(first, N):
        imaging = i >= 0
        phase = np.mod(117 * (i**2 + i + 2), 360) * np.pi / 180 if spoil else 0.0
        rf.phase_offset = phase
        adc.phase_offset = phase
        area_y = phase_areas[i] if imaging else 0.0
        gy_pre = pp.make_trapezoid('y', area=area_y, duration=1e-3, system=system)

        seq.add_block(rf, gz)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(delay_block(delay_te))
        if imaging:
            seq.add_block(gx, adc)
        else:
            seq.add_block(gx)
        if spoil:
            gy_rewind = pp.make_trapezoid('y', area=-area_y, duration=1e-3, system=system)
            seq.add_block(gx_spoil, gy_rewind, gz_spoil)
        seq.add_block(delay_block(delay_tr))
    return seq


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

def offres_phantom(dfs, T2):
    """ MRZero phantom: one isochromat at the origin per off-resonance in dfs (Hz).
        T1 is huge, so the only decay is T2 (s) and the off-resonance fan. """
    n = len(dfs)
    phantom = mr0.CustomVoxelPhantom(
        pos=np.zeros((n, 3), dtype=np.float32), PD=1.0, T1=1e9, T2=T2, T2dash=1e9,
        D=0.0, B0=np.asarray(dfs, dtype=np.float32), B1=1.0, voxel_size=1e-3,
    )
    # MRZero's signal is complex64, so every map has to be float32.
    for name in ('PD', 'T1', 'T2', 'T2dash', 'D', 'B0', 'B1', 'voxel_pos', 'voxel_size'):
        setattr(phantom, name, getattr(phantom, name).float())
    return phantom


def simulate_seq(seq, phantom=None, size=(64, 64)):
    """ simulate seq with MRZero. phantom=None uses MRZero's brain at size.
        returns signal (n_samples,) complex and ktraj (3, n_samples) in 1/m. """
    with _quietly_in_cache():
        signal, ktraj = mr0.util.simulate(seq, phantom=phantom,
                                          sim_size=size if phantom is None else None)
    signal = signal.numpy().sum(axis=-1)  # sum over receive coils
    return signal, ktraj.numpy()[:, :3].T


def simulate_isochromats(seq, dfs, T2=1e9):
    """ simulate each off-resonance in dfs on its own. one isochromat's signal
        is just its own transverse magnetization, so this returns mxy
        (n_spins, n_samples): one arrow per spin, at every ADC sample. """
    return np.stack([simulate_seq(seq, offres_phantom([df], T2))[0] for df in dfs])


def brain_maps(size=(64, 64)):
    """ PD, T1 (s), and T2 (s) of MRZero's brain, each (size) for imshow. """
    with _quietly_in_cache():
        phantom = mr0.util.load_phantom(size)
    return [np.rot90(m[:, :, 0].numpy()) for m in (phantom.PD, phantom.T1, phantom.T2)]
