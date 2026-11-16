#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Simple Example that runs the MSO model for 400Hz with a 400 Hz
tone amplitude modulated at 2Hz and plots the firing rate difference.

The expensive simulation result is cached to calc_amp_mod_data.npz so you
can iterate on the plot without re-running the simulation:
    python examples/calc_amp_mod.py          # run sim (+ cache) then plot
    python examples/calc_amp_mod.py --plot   # load cache, plot only
    python examples/calc_amp_mod.py --force  # ignore cache, re-run sim
'''

from __future__ import print_function, division

import matplotlib
matplotlib.use('Agg')

import os
import sys

import numpy as np
import brian as br
from brian.units import second
import cochlea as coch
import thorns

from mso_model.mso_connect import inhibitory_to_mso, excitatory_to_mso, cochlea_to_gbc
from mso_model.mso import make_mso_group
from mso_model.helper import run
import mso_model.audiotools as audio

import matplotlib.pyplot as plt

CACHE = 'calc_amp_mod_data.npz'


def simulate(c_freq=400, sound_freq=400, itd=200e-6, n_neuron=300):
    """Run the full MSO simulation.

    Returns
    -------
    (spike_times_left, spike_times_right, duration)
        Each is a flat numpy array of spike times in seconds.
    """
    br.defaultclock.dt = 20E-6 * second

    # Basic parameters
    fs_coch = 100e3            # samples / second
    duration = 1800E-3         # seconds
    pad = 120E-3               # seconds
    dt_coch = 1 / fs_coch
    n_pad = int(pad / dt_coch)
    const_dt = 100e-6

    sound = audio.generate_tone(sound_freq, duration, fs_coch)
    mod = audio.cos_amp_modulator(sound, 2, fs_coch)
    sound *= mod
    sound = coch.set_dbspl(sound, 50)
    sound = sound * audio.cosine_fade_window(sound, 20e-3, fs_coch)
    sound = np.concatenate((np.zeros(n_pad), sound, np.zeros(n_pad)))

    sound = audio.delay_signal(sound, np.abs(itd), fs_coch)

    duration = len(sound) / fs_coch

    # Construct ipsi- and contra-lateral ANF trains and convert to groups.
    cochlea_train_left = coch.run_zilany2014(sound=sound[:, 0],
                                             fs=fs_coch,
                                             anf_num=(n_neuron, 0, 0),
                                             cf=c_freq,
                                             species='human',
                                             seed=None)

    cochlea_train_right = coch.run_zilany2014(sound=sound[:, 1],
                                             fs=fs_coch,
                                             anf_num=(n_neuron, 0, 0),
                                             cf=c_freq,
                                             species='human',
                                             seed=None)

    anf_group_left = coch.make_brian_group(cochlea_train_left)
    anf_group_right = coch.make_brian_group(cochlea_train_right)

    # Setup MSO and GBC groups.
    mso_group_left = make_mso_group(n_neuron)
    mso_group_right = make_mso_group(n_neuron)

    gbc_group_left = cochlea_to_gbc(anf_group_left, n_neuron)
    gbc_group_right = cochlea_to_gbc(anf_group_right, n_neuron)

    # Inhibitory connections.
    syn_mso_l_in_ipsi, syn_mso_l_in_contra = inhibitory_to_mso(
        mso_group=mso_group_left,
        ipsi_group=gbc_group_left['neuron_groups'][0],
        contra_group=gbc_group_right['neuron_groups'][0],
        strength=1.5e-8,
        ipsi_delay=0,
        contra_delay=-0.6e-3 + const_dt)

    syn_mso_r_in_ipsi, syn_mso_r_in_contra = inhibitory_to_mso(
        mso_group=mso_group_right,
        ipsi_group=gbc_group_right['neuron_groups'][0],
        contra_group=gbc_group_left['neuron_groups'][0],
        strength=1.5e-8,
        ipsi_delay=0,
        contra_delay=-0.6e-3 + const_dt)

    # Excitatory connections.
    syn_mso_l_ex_ipsi, syn_mso_l_ex_contra = excitatory_to_mso(
        mso_group=mso_group_left,
        ipsi_group=anf_group_left,
        contra_group=anf_group_right,
        strength=20e-9,
        contra_delay=const_dt)

    syn_mso_r_ex_ipsi, syn_mso_r_ex_contra = excitatory_to_mso(
        mso_group=mso_group_right,
        ipsi_group=anf_group_right,
        contra_group=anf_group_left,
        strength=20e-9,
        contra_delay=const_dt)

    sp_mon_left = br.SpikeMonitor(mso_group_left, record=True)
    sp_mon_right = br.SpikeMonitor(mso_group_right, record=True)

    network = ([mso_group_left, mso_group_right,
                anf_group_left, anf_group_right,
                syn_mso_l_ex_ipsi, syn_mso_l_ex_contra,
                syn_mso_l_in_ipsi, syn_mso_l_in_contra,
                syn_mso_r_ex_ipsi, syn_mso_r_ex_contra,
                syn_mso_r_in_ipsi, syn_mso_r_in_contra,
                sp_mon_left, sp_mon_right]
               + gbc_group_left['neuron_groups']
               + gbc_group_right['neuron_groups'])

    run(duration, network)

    # Brian 1 SpikeMonitor stores flat spike times (in seconds) in .spikes.
    spike_times_left = np.asarray(sp_mon_left.spikes, dtype=float)
    spike_times_right = np.asarray(sp_mon_right.spikes, dtype=float)

    return spike_times_left, spike_times_right, float(duration)


def save_cache(path, left, right, duration):
    np.savez(path,
             spike_times_left=left,
             spike_times_right=right,
             duration=duration)
    print('Saved simulation result to %s' % path)


def load_cache(path):
    with np.load(path) as data:
        return (data['spike_times_left'],
                data['spike_times_right'],
                float(data['duration']))


def plot(left, right, duration, bin_width=30e-3):
    """Plot the firing-rate difference (contra - ipsi) as a function of time."""
    # Trim the silent padding at the start/stop (same as before: 0.125 .. 2.125 s).
    start, stop = 0.125, 2.125
    left = left[(left >= start) & (left <= stop)]
    right = right[(right >= start) & (right <= stop)]

    # Plain-numpy PSTH (rate in spikes/sec per bin).
    bins = np.arange(start, stop + bin_width, bin_width)
    rate_l, _ = np.histogram(left, bins=bins)
    rate_r, _ = np.histogram(right, bins=bins)
    rate_l = rate_l.astype(float) / bin_width
    rate_r = rate_r.astype(float) / bin_width

    fig, ax = plt.subplots(1, 1)
    ax.plot(bins[:-1], rate_r - rate_l)
    ax.set_xlabel('time / s')
    ax.set_ylabel('$\Delta R$ / sps')
    fig.tight_layout()
    fig.savefig('calc_amp_mod.png', dpi=150)
    print('Saved plot to calc_amp_mod.png')


def main():
    force = '--force' in sys.argv
    plot_only = '--plot' in sys.argv

    if (not force) and os.path.exists(CACHE):
        print('Found cache %s; skipping simulation (use --force to re-run).'
              % CACHE)
        left, right, duration = load_cache(CACHE)
    else:
        if plot_only:
            print('No cache found at %s; running simulation first.' % CACHE)
        left, right, duration = simulate()

    # Only write the cache if we actually simulated (or forced it).
    if force or not os.path.exists(CACHE):
        save_cache(CACHE, left, right, duration)

    plot(left, right, duration)


if __name__ == '__main__':
    main()
