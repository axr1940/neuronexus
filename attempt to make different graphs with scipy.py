import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from radiens import VidereClient

# THE REAL GOLDEN PATH
full_path = r"C:\Users\axr1940\radix\data\allego_1__uid0331-10-58-20.xdat"

def main():
    print("Connecting to Videre and loading data...")
    vc = VidereClient()
    
    # 1. LOAD THE METADATA
    meta = vc.link_data_file(full_path)
    print("Data successfully linked! The bouncer let us in!")

    # 2. PULL THE SIGNALS SEPARATELY (No more "all"!)
    print("Asking the bartender for brain waves (amp)...")
    brain_sigs = vc.signals().get_signals(
        dataset_metadata=meta,
        sig_type="amp",  
        ntv_idxs="all"
    )
    
    print("Asking the bartender for digital beeps (gpio_dout)...")
    beep_sigs = vc.signals().get_signals(
        dataset_metadata=meta,
        sig_type="dout",  
        ntv_idxs="all"
    )
    print("All data successfully loaded into memory!")

    # 3. SET UP THE SLICER PARAMETERS
    sampling_rate = 30000 
    pre_samples = int(0.050 * sampling_rate)   # 50 ms before 
    post_samples = int(0.150 * sampling_rate)  # 150 ms after 

    # 4. GRAB THE DIGITAL BEEPS (DO 1)
    digital_out = beep_sigs.signals.gpio_dout[0] 
    
    # Find where the beep goes from 0 to 1
    trigger_samples = np.where(np.diff(digital_out) > 0)[0] 
    print(f"Success: I found {len(trigger_samples)} beeps in the file!")

    # 5. SLICE THE BRAIN WAVES (Electrode 1)
    brain_data_ch1 = brain_sigs.signals.amp[0] 
    all_slices = [] 

    for beep_index in trigger_samples:
        start_point = beep_index - pre_samples
        end_point = beep_index + post_samples
        
        # Keep slices that fit perfectly inside the recording time
        if start_point > 0 and end_point < len(brain_data_ch1):
            one_slice = brain_data_ch1[start_point : end_point]
            all_slices.append(one_slice)

    print(f"Successfully created {len(all_slices)} epochs (slices) of data!")
# --- THE VICTORY LAP: PLOTTING THE DATA ---
    print("Crunching the numbers and drawing the graph...")
    
    # 1. Average all 10 slices together to find the clear signal
    average_brain_wave = np.mean(all_slices, axis=0)

    # 2. Create a time axis in milliseconds (-50ms to +150ms)
    time_axis = np.linspace(-50, 150, len(average_brain_wave))

    # 3. Draw the graph!
    plt.figure(figsize=(10, 5)) # Make it nice and wide
    plt.plot(time_axis, average_brain_wave, color='blue', linewidth=2)
    
    # Draw a red dashed line exactly where the beep happened (Time = 0)
    plt.axvline(x=0, color='red', linestyle='--', label='Stimulus (Beep)') 
    
    plt.title("Brain's Average Reaction to the Beep (ERP)")
    plt.xlabel("Time (milliseconds)")
    plt.ylabel("Voltage")
    plt.legend()
    plt.grid(True, alpha=0.3) # Adds a faint grid for readability
    
    # This command physically pops the window open on your screen
    plt.show()
    # --- STEP 2: THE FFT (FREQUENCY ANALYSIS) ---
    print("Calculating the FFT to find the carrier frequency...")
    
    # 1. Run the FFT math on our averaged brain wave
    fft_values = np.fft.fft(average_brain_wave)
    
    # 2. Figure out the X-axis (the actual frequencies in Hz)
    n_samples = len(average_brain_wave)
    frequencies = np.fft.fftfreq(n_samples, d=1/sampling_rate)
    
    # 3. We only care about the positive frequencies (the first half)
    half_n = n_samples // 2
    pos_frequencies = frequencies[:half_n]
    pos_fft_values = np.abs(fft_values[:half_n]) # Get the power/magnitude
    
    # 4. Draw the FFT Graph!
    plt.figure(figsize=(10, 5))
    plt.plot(pos_frequencies, pos_fft_values, color='purple', linewidth=1.5)
    
    plt.title("FFT: What frequencies are hiding in the brain wave?")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Power)")
    
    # Let's zoom in on the 0 to 5000 Hz range (Change 5000 if your beep was higher pitch!)
    plt.xlim(0, 5000) 
    plt.grid(True, alpha=0.3)
    
    plt.show()

    # =======================================================
    # --- LEVEL 3: EXTRACTING ACTION POTENTIALS (SPIKES) ---
    # =======================================================
    print("Designing the Bandpass Filter (300 - 3000 Hz)...")

    # 1. Set up the math for the neuro-filter
    # The Nyquist rate is exactly half of your sampling rate (30000 / 2 = 15000)
    nyq = 0.5 * sampling_rate
    low_cut = 300 / nyq
    high_cut = 3000 / nyq

    # Create a 4th-order Butterworth bandpass filter
    b, a = butter(4, [low_cut, high_cut], btype='bandpass')

    # 2. Grab ONE continuous channel of your raw brain data
    # (Assuming you already have a variable like 'brain_data_ch1' from earlier)
    print("Scrubbing away the slow waves to reveal the spikes...")
    filtered_spikes = filtfilt(b, a, brain_data_ch1)

    # 3. Let's look at a random 100-millisecond window of continuous data 
    # (e.g., from exactly 1.0 to 1.1 seconds into the recording)
    start_idx = int(1.0 * sampling_rate)
    end_idx = int(1.1 * sampling_rate)

    time_snippet = np.linspace(0, 100, end_idx - start_idx) # Time in ms
    raw_snippet = brain_data_ch1[start_idx:end_idx]
    filt_snippet = filtered_spikes[start_idx:end_idx]

    # 4. Draw the Before & After Graphs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top Graph: The Raw Data (Notice the slow rolling waves)
    ax1.plot(time_snippet, raw_snippet, color='gray', linewidth=1)
    ax1.set_title("Raw Wideband Signal (LFPs + Spikes)")
    ax1.set_ylabel("Voltage")
    ax1.grid(True, alpha=0.3)

    # Bottom Graph: The Filtered Data (Only Action Potentials!)
    ax2.plot(time_snippet, filt_snippet, color='black', linewidth=1)
    
    # Let's draw a mock "Spike Detection Threshold" line
    # (Action potentials usually shoot downwards first in extracellular recordings)
    threshold = -30 # You may need to tweak this number depending on your signal scale!
    ax2.axhline(threshold, color='red', linestyle='--', label=f'Spike Detection Threshold')
    
    ax2.set_title("Bandpass Filtered (300-3000 Hz) - Action Potentials Revealed!")
    ax2.set_xlabel("Time (milliseconds)")
    ax2.set_ylabel("Voltage")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()