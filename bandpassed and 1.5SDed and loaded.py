import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
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
# Let's see what channels are inside this file!
    print("\n--- AVAILABLE CHANNELS ---")
    
    # If your data object has a method to get channel names, we will print them.
    # (I'm using 'dir()' here so Python literally lists out everything the object can do)
    print("What can this data object do?:", dir(data)) 
    
    # NOTE: If you have a variable holding the raw data, try printing its keys or shape!
    print("--------------------------\n")
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
    # --- LEVEL 3: PAPER-REPLICATED SPIKE EXTRACTION ---
    # =======================================================
    print("Designing the Bandpass Filter (100 - 3000 Hz)...")

    # 1. Update filter to match the paper (100 to 3000 Hz)
    nyq = 0.5 * sampling_rate
    low_cut = 100 / nyq
    high_cut = 3000 / nyq
    b, a = butter(4, [low_cut, high_cut], btype='bandpass')

    print("Filtering data...")
    filtered_spikes = filtfilt(b, a, brain_data_ch1)

    # 2. Calculate the Dynamic Threshold (1.5x Standard Deviation)
    # We calculate the SD of the whole filtered signal to find the baseline noise
    noise_sd = np.std(filtered_spikes)
    
    # Action potentials usually shoot downwards first, so our threshold is negative
    spike_threshold = -1.5 * noise_sd 
    print(f"Calculated 1.5x SD Threshold: {spike_threshold:.2f} uV")

    # 3. Find the Spikes!
    # We flip the signal negative (-) so the "valleys" become "peaks" for the algorithm to find.
    # The 'distance' parameter ensures we don't count a single wide spike twice (e.g., locking out for 1 ms)
    lockout_ms = int(0.001 * sampling_rate) 
    spike_indices, _ = find_peaks(-filtered_spikes, height=-spike_threshold, distance=lockout_ms)
    
    print(f"SUCCESS: Found {len(spike_indices)} total spikes in the recording!")

    # 4. Plotting a snippet to verify
    start_idx = int(1.0 * sampling_rate)
    end_idx = int(1.1 * sampling_rate)
    
    # Find which spikes specifically live inside our 100ms viewing window
    spikes_in_window = [s for s in spike_indices if start_idx <= s < end_idx]
    
    time_snippet = np.linspace(0, 100, end_idx - start_idx)
    filt_snippet = filtered_spikes[start_idx:end_idx]

    plt.figure(figsize=(12, 5))
    plt.plot(time_snippet, filt_snippet, color='black', linewidth=1, label="Filtered Data (100-3000 Hz)")
    plt.axhline(spike_threshold, color='red', linestyle='--', label=f'Threshold (1.5x SD = {spike_threshold:.1f})')
    
    # Put a distinct dot exactly where the computer detected a spike
    for spike_idx in spikes_in_window:
        # Calculate where on our 100ms X-axis this spike belongs
        spike_time_ms = ((spike_idx - start_idx) / sampling_rate) * 1000
        plt.plot(spike_time_ms, filtered_spikes[spike_idx], 'ro') # 'ro' means Red Circle

    plt.title("Automated Spike Detection (Paper Methodology)")
    plt.xlabel("Time (milliseconds)")
    plt.ylabel("Voltage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # =======================================================
    # --- LEVEL 4: PERI-STIMULUS TIME HISTOGRAM (PSTH) ---
    # =======================================================
    print("Building the PSTH...")

    # 1. The Triggers (When did the speaker fire?)
    # For now, let's pretend the speaker fired at these exact milliseconds:
    trigger_times_ms = np.array([500, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500])
    
    # 2. Set our window (How much time around the beep do we care about?)
    pre_stim_ms = 50   # Look 50ms before the beep
    post_stim_ms = 200 # Look 200ms after the beep
    bin_size_ms = 5    # Group spikes into 5ms "buckets" for the bar chart

    # 3. Align the Spikes!
    aligned_spikes = []
    raster_y_coords = []
    raster_x_coords = []

    # Loop through every single time the speaker fired (every trial)
    for trial_number, trigger_time in enumerate(trigger_times_ms):
        
        # Define the absolute start and end time for this specific trial's window
        window_start = trigger_time - pre_stim_ms
        window_end = trigger_time + post_stim_ms
        
        # Find all spikes that happened inside this window
        # (Assuming you still have 'spike_times_ms' from the previous code block)
        spikes_in_window = [s for s in spike_times_ms if window_start <= s <= window_end]
        
        # Align them so the trigger is exactly at Time = 0
        for spike_time in spikes_in_window:
            relative_time = spike_time - trigger_time
            aligned_spikes.append(relative_time)
            
            # Save coordinates for the Raster Plot
            raster_x_coords.append(relative_time)
            raster_y_coords.append(trial_number + 1)

    # 4. Draw the Publication-Ready Figure (Raster + PSTH)
    fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    # Top Graph: Raster Plot (Every dot is a single spike across trials)
    ax_raster.scatter(raster_x_coords, raster_y_coords, color='black', s=10, marker='|')
    ax_raster.axvline(0, color='red', linestyle='--', linewidth=1)
    ax_raster.set_title("Neural Firing Aligned to Stimulus")
    ax_raster.set_ylabel("Trial Number")
    ax_raster.set_yticks(range(1, len(trigger_times_ms) + 1))
    
    # Bottom Graph: The PSTH (Bar chart of total spikes)
    bins = np.arange(-pre_stim_ms, post_stim_ms + bin_size_ms, bin_size_ms)
    ax_psth.hist(aligned_spikes, bins=bins, color='gray', edgecolor='black')
    ax_psth.axvline(0, color='red', linestyle='--', linewidth=2, label="Stimulus Onset")
    
    ax_psth.set_xlabel("Time from Stimulus (milliseconds)")
    ax_psth.set_ylabel("Total Spike Count")
    ax_psth.legend()
    
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()