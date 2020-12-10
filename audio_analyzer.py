from IPython.display import Audio, Image,HTML
from IPython import display
from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal
import time

def high_pass_filter(audio_data, sample_rate=44100, min_f = 0, max_f=20000):
    freqs = np.fft.rfftfreq(len(audio_data), d=1.0/sample_rate)
    audio_fft = np.fft.rfft(audio_data)
    audio_fft[freqs < min_f] = 0
    audio_fft[freqs > max_f] = 0
    return np.fft.irfft(audio_fft)

def snippet(audio_data, ts, duration, percent_max=6):
    min_ts = ts[audio_data < np.max(audio_data)]
    max_t_loud = max(min_ts[min])

def sample_clip(audio_data, ts, t0=None, t1=None):
    if t0 is None:
        t0 = np.min(ts)
    if t1 is None:
        t1 = np.max(ts)
        
    audio_data = audio_data[np.nonzero(t0<=ts)]
    ts = ts[np.nonzero(t0<=ts)]
    
    audio_data = audio_data[np.nonzero(ts<=t1)]
    ts = ts[np.nonzero(ts<=t1)]
    
    return audio_data, ts

def find_peaks_windowed(signal, window, rel_height=0.2, distance=5, **kwargs):
    rolled_sig = np.empty((window, len(signal)))
    rolled_sig[:] = signal
    for i in range(window):
        rolled_sig[i,:] = np.roll(rolled_sig[i, :], i)
    
    meds = np.median(rolled_sig, axis=0) 
    
    height = rel_height * np.max(signal - meds)
    max_indices, peak_data = scipy.signal.find_peaks(signal - meds, height=height, distance=distance, **kwargs)
    
    return max_indices
    
def analyze_sound_file(audio_file, t0=None, t1=None, window_in_hz=300, distance_in_hz=100, rel_height=0.2):
    sampling_rate, ys = wavfile.read(audio_file)
    
    # collapse to mono
    if len(ys.shape) == 2:
        ys.sum(axis=1)
    
    ts = np.linspace(0, len(ys) / sampling_rate, len(ys))
    ys, ts = sample_clip(ys, ts, t0, t1)

    freqs = np.fft.rfftfreq(len(ys), d=1.0/sampling_rate)
    freq_bin = 1 / (1.0/sampling_rate * len(ys))
    
    window = int(window_in_hz / freq_bin)
    
    distance = int(distance_in_hz / freq_bin)
    
    audio_fft = np.fft.rfft(ys)
    power_spectrum = np.log(np.real(audio_fft * audio_fft.conj()))
    
    max_indices = find_peaks_windowed(
        signal=power_spectrum,
        window=window,
        distance=distance,
	rel_height=rel_height,
    )
    
    max_indices = max_indices[np.nonzero(freqs[max_indices] > 400)]
    
    amplitudes = np.sqrt(np.exp(power_spectrum[max_indices]))

    phases = np.mod(np.arctan2(
        np.imag(audio_fft),
        np.real(audio_fft),
    ) + 2 * np.pi, 2 * np.pi)
    
    period = 1.0/freqs[max_indices][0]
    
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ts, ys, '-k')
    ax1.set_title('Source Soundwave')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(t0, t0+3*period)
    
    ys_synth, ts_synth = generate_waveform_from_sinewaves(
        frequencies=freqs[max_indices],
        amplitudes=amplitudes, 
        phases=phases[max_indices],
        sampling_rate=44100
    )
    
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(ts_synth, ys_synth, '-k')  
    ax4.set_xlim(t0, t1)
    ax4.set_title('Synthesized Soundwave')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.set_xlim(0, 3*period)
    
    return {
        'synth': {
            'audio_data': ys_synth,
            'sampling_rate': 44100
        },
        'analysis': {
            'amplitudes': amplitudes,
            'frequencies': freqs[max_indices],
            'phases': phases[max_indices]
        }
    }

def get_synth_and_source_audio_html(instrument, source_audio_file, synth_audio_file):
    audio_elem = HTML(
        '''
        <style>
            .playback_container {{
                background-color:#FFFFFF;
                float:left
            }}
            
            .helper {{
                display: inline-block;
                height: 100%;
                vertical-align: middle;
            }}

            .instr_image {{
                background: #3A6F9A;
                vertical-align: middle;
                max-height: 150px;
                max-width: 230px;
            }}
            
            .instrument {{
                height: 150px;      /* Equals maximum image height */
                border: 1px solid red;
                white-space: nowrap; /* This is required unless you put the helper span closely near the img */

                text-align: center;
                margin: 1em 0;
            }}
            
        </style>
        <div>
            <div width=10% class=playback_container>
                <figure>
                    <figcaption> {instrument:} </figcaption>
                    <audio controls src={source_audio_file}></audio>
                </figure>
                  <figure>
                    <figcaption> Synthesizer </figcaption>
                    <audio controls src={synth_audio_file:}></audio>
                </figure>
            </div>
            <div width=100%, class=frame>
                <span class=helper/>
                <img class=instr_image src=images/{instrument:}.jpg height=100%/>
            </div>
        </div>
        '''.format(
            instrument=instrument,
            source_audio_file=source_audio_file,
            synth_audio_file=synth_audio_file,
        )
    )
    return audio_elem

def display_synth_and_source_audio_html(instrument, sample_audio_file, synth_audio_file):
    elem = get_synth_and_source_audio_html(instrument, sample_audio_file, synth_audio_file)
    display.display(elem)
    
def generate_waveform_from_sinewaves(
        amplitudes,
        frequencies,
        phases=None,
        duration=5,
        sampling_rate=44100
    ):
        ys = np.zeros(duration * sampling_rate, dtype=np.float32)
        ts = np.linspace(0, duration, duration * sampling_rate)
        if phases is None:
            phases = np.zeros(len(amplitudes))
        for f, a, p in zip(frequencies, amplitudes, phases):
            ys += a * np.cos(2 * np.pi * f * ts + p)

        ys /= np.max(ys)
        return ys, ts
    
def generate_square_wave(
        frequency,
        ts
    ):
        period = 1.0/frequency
        
        ys = np.ones(ts.shape)
        ys[np.nonzero(np.abs(np.remainder(ts, period)) > 0.5 * period)] = -1
        return ys
    
def generate_sawtooth_wave(
        frequency,
        ts,
    ):
        return scipy.signal.sawtooth(2 * np.pi * frequency * ts)

def generate_phase_sweep(frequency, ts, phi_max=8*np.pi, a1=1, a2=2):
    phase_sweep = np.linspace(0, phi_max, len(ts))
    ys = a1 * np.cos(2 * np.pi * frequency * ts)
    ys += a2 * np.cos(2 * np.pi * frequency * ts + phase_sweep)
    return ys

def generate_beats(frequency, ts, delta_f=5):
    y1s = np.cos(2 * np.pi * frequency * ts)
    y2s = np.cos((2 * np.pi * (frequency + delta_f) )* ts)
    return y1s, y2s, y1s + y2s

def show_signal_buildup_from_components(
        amplitudes,
        frequencies,
        phases,
        filename_base='sample',
        duration=5,
        sampling_rate=44100,
        origin_power_spectrum=None
    ):
    
    html_template =  '''
        <style>
            .playback_container {{
                background-color:#FFFFFF;
                float:left
            }}
            
            .helper {{
                display: inline-block;
                height: 100%;
                vertical-align: middle;
            }}

            .plot_image {{
                background: #3A6F9A;
                vertical-align: middle;
                max-height: 150px;
                max-width: 300px;
            }}
            
            .plot {{
                height: 150px;      /* Equals maximum image height */
                border: 1px solid red;
                white-space: nowrap; /* This is required unless you put the helper span closely near the img */
                text-align: center;
                margin: 1em 0;
            }}
            
        </style>
        <div>
            {rows:}
        </div>
    '''
        
    html_row = '''
        <div>
            <div width=10% class=playback_container>
                <figure>
                    <figcaption> {num_components} Frequency Components </figcaption>
                    <audio controls src={audio_file:}?{timestamp:}></audio>
                </figure>
            </div>
            <div width=100%, class=frame>
                <span class=helper/>
                <img class=plot_image src={plot_file:}?{timestamp:} height=100%/>
            </div>
        </div>
    '''
    
    html_rows = list()
    if len(amplitudes) > 8:
        num_comps = [1, 2, 4, 8, len(amplitudes)]
    else:
        num_comps = [int(i) for i in np.round(np.linspace(1, len(amplitudes), min(len(amplitudes), 5)))]
    period = 1.0 / frequencies[0]
    plt.ioff()
    normalized_amplitudes = np.array(amplitudes)/np.max(amplitudes)

    for i, n in enumerate(num_comps):
        
        fig = plt.figure(figsize=(10, 2))
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        
        for j in range(n):
            _ys, _ts = generate_waveform_from_sinewaves(
                [amplitudes[j]],
                [frequencies[j]],
                [phases[j]],
                duration=duration,
                sampling_rate=44100
            )
            ax1.plot(_ts, _ys * normalized_amplitudes[j], '--', alpha=0.5)

        ys, ts = generate_waveform_from_sinewaves(
            amplitudes[:n],
            frequencies[:n],
            phases[:n],
            duration=duration,
            sampling_rate=44100
        )
        ax1.plot(ts, ys, '-k')

        ax1.set_title('{} Frequency Components'.format(n))
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlim(0, 4*period)
        
        sound_filename = 'samples/{}_{}.wav'.format(filename_base, i)
        image_filename = 'images/{}_{}.png'.format(filename_base, i)
        
        fig.savefig(image_filename)
        plt.close(fig)
        with open(sound_filename, 'wb') as outf:
            ys = ys / np.max(ys)

            wavfile.write(
                outf,
                data=ys.astype(np.float32),
                rate=sampling_rate,
            )
        html_rows.append(html_row.format(
            plot_file=image_filename,
            audio_file=sound_filename,
            num_components=n,
            timestamp=time.time()
        ))

    plt.ion()
    display.display(HTML(
        html_template.format(rows='\n'.join(html_rows))
    ))


def full_analysis(selected_instrument, student_name, t0=0, t1=None, rel_height=0.2):
    source_filename = 'samples/{}a4.wav'.format(selected_instrument)

    results = analyze_sound_file(
        source_filename,
        t0=t0,
        t1=t1,
        distance_in_hz=400,
	rel_height=rel_height,
    )

    synth_filename = 'samples/{}_{}_full_synth.wav'.format(selected_instrument, student_name)
    with open(synth_filename, 'wb') as outf:
        ys = results['synth']['audio_data'].astype(np.float32)
        ys = ys / np.max(ys)

        wavfile.write(
            outf,
            data=ys,
            rate=results['synth']['sampling_rate'],
        )

    display_synth_and_source_audio_html(
        sample_audio_file=source_filename,
        synth_audio_file=synth_filename,
        instrument=selected_instrument
    )
    show_signal_buildup_from_components(
        **results['analysis'],
        filename_base='_'.join([student_name, selected_instrument])
    )

def fourier_analysis(selected_instrument, t0=0, t1=None, rel_height=0.2):
    source_filename = 'samples/{}a4.wav'.format(selected_instrument)

    results = analyze_sound_file(
        source_filename,
        t0=t0,
        t1=t1,
        distance_in_hz=400,
	rel_height=rel_height,
    )

    synth_filename = 'samples/{}_full_synth.wav'.format(selected_instrument)
    with open(synth_filename, 'wb') as outf:
        ys = results['synth']['audio_data'].astype(np.float32)
        ys = ys / np.max(ys)

        wavfile.write(
            outf,
            data=ys,
            rate=results['synth']['sampling_rate'],
        )

    display_synth_and_source_audio_html(
        sample_audio_file=source_filename,
        synth_audio_file=synth_filename,
        instrument=selected_instrument
    )
    show_signal_buildup_from_components(
        **results['analysis'],
        filename_base='_'.join(['synth', selected_instrument])
    )


def generate_sine_wave(t0, t1, y0, y1):
    ys, ts = generate_waveform_from_sinewaves(
        amplitudes=[1],
        frequencies=[440],
        phases=None,
        duration=5,
        sampling_rate=44100
    )
    outf_name = 'samples/s1.wav'

    ys = ys.astype(np.float32)
    ys = ys / np.max(ys)
    wavfile.write(
        filename=outf_name,
        data=ys,
        rate=44100
    )

    plt.figure()
    plt.plot(ts, ys, '-k')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.ylim(y0, y1)
    plt.xlim(t0, t1)
    plt.show()
    display.display(Audio(outf_name))

def generate_a_different_sine_wave(t0, t1, y0, y1):
    ys, ts = generate_waveform_from_sinewaves(
        amplitudes=[1],
        frequencies=[587.33],
        phases=None,
        duration=5,
        sampling_rate=44100
    )
    outf_name = 'samples/s1.wav'

    ys = ys.astype(np.float32)
    ys = ys / np.max(ys)
    wavfile.write(
        filename=outf_name,
        data=ys,
        rate=44100
    )

    plt.figure()
    plt.plot(ts, ys, '-k')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.ylim(y0, y1)
    plt.xlim(t0, t1)
    plt.show()
    display.display(Audio(outf_name))

def generate_two_sine_waves_with_different_frequencies(t0, t1, y0, y1):
    y1s, ts = generate_waveform_from_sinewaves(
        amplitudes=[1],
        frequencies=[440],
        phases=None,
        duration=5,
        sampling_rate=44100
    )

    y2s, ts = generate_waveform_from_sinewaves(
        amplitudes=[1],
        frequencies=[587.33],
        phases=None,
        duration=5,
        sampling_rate=44100
    )

    outf_name = 'samples/s2.wav'

    ys = (y1s + y2s).astype(np.float32)
    y1s = y1s / np.max(ys)
    y2s = y2s / np.max(ys)
    ys = ys / np.max(ys)
    wavfile.write(
        filename=outf_name,
        data=ys,
        rate=44100
    )

    plt.figure()
    plt.plot(ts, ys, '-k')
    plt.plot(ts, y1s, '--r', label='y1')
    plt.plot(ts, y2s, '--b', label='y2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.ylim(y0, y1)
    plt.xlim(t0, t1)
    plt.show()
    display.display(Audio(outf_name))


def generate_two_sine_waves_with_different_phases(t0, t1, y0, y1):
    y1s, ts = generate_waveform_from_sinewaves(
        amplitudes=[1],
        frequencies=[440],
        phases=None,
        duration=5,
        sampling_rate=44100
    )

    y2s, ts = generate_waveform_from_sinewaves(
        amplitudes=[1],
        frequencies=[440],
        phases=[np.pi * 0.25],
        duration=5,
        sampling_rate=44100
    )

    outf_name = 'samples/s3.wav'

    ys = (y1s + y2s).astype(np.float32)
    y1s = y1s / np.max(ys)
    y2s = y2s / np.max(ys)
    ys = ys / np.max(ys)
    wavfile.write(
        filename=outf_name,
        data=ys,
        rate=44100
    )

    plt.figure()
    plt.plot(ts, ys, '-k')
    plt.plot(ts, y1s, '--r', label='y1')
    plt.plot(ts, y2s, '--b', label='y2')
    plt.ylim(y0, y1)
    plt.xlim(t0, t1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    display.display(Audio(outf_name))


def generate_three_sine_waves_with_different_frequencies():
    y1s, ts = generate_waveform_from_sinewaves(
        amplitudes=[1],
        frequencies=[440],
        phases=None,
        duration=5,
        sampling_rate=44100
    )

    y2s, ts = generate_waveform_from_sinewaves(
        amplitudes=[1],
        frequencies=[554.37],
        phases=None,
        duration=5,
        sampling_rate=44100
    )

    y3s, ts = generate_waveform_from_sinewaves(
        amplitudes=[1],
        frequencies=[659.25],
        phases=None,
        duration=5,
        sampling_rate=44100
    )

    outf_name = 'samples/s4.wav'

    ys = (y1s + y2s + y3s).astype(np.float32)
    y1s = y1s / np.max(ys)
    y2s = y2s / np.max(ys)
    y3s = y3s / np.max(ys)

    ys = ys / np.max(ys)
    wavfile.write(
        filename=outf_name,
        data=ys,
        rate=44100
    )

    plt.figure()
    plt.plot(ts, ys, '-k')
    plt.plot(ts, y1s, '--r', label='y1')
    plt.plot(ts, y2s, '--b', label='y2')
    plt.plot(ts, y3s, '--g', label='y3')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    display.display(Audio(outf_name))
