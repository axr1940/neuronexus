import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from radiens import VidereClient

# --- CONFIGURATION ---
full_path = r"C:\Users\axr1940\radix\data\allego_7__uid0519-12-50-43.xdat"
sampling_rate = 30000 
target_channel = 1        # The "Busy" Channel we found
threshold_multiplier = 3.5 # High enough to isolate the clean unit

# Windows
t_start, t_end = 115, 140  # Wide view for context
bbn_start, bbn_end = 123, 131 # Your known BBN window

def master_dcn_analysis():
    vc = VidereClient()
    meta = vc.link_data_file(full_path)
    
    # 1. LOAD & FILTER
    print(f"Loading Channel {target_channel}...")
    sigs = vc.signals().get_signals(dataset_metadata=meta, sig_type="amp", ntv_idxs=[target_channel], time_range=[t_start, t_end])
    raw = sigs.signals.amp[0]
    
    nyquist = 0.5 * sampling_rate
    b, a = signal.butter(4, [100/nyquist, 3000/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, raw)
    
    # 2. DETECT SPIKES
    thresh = threshold_multiplier * np.std(filtered)
    spike_idxs = np.where(filtered < -thresh)[0]
    
    # 3. CLEAN UP & EXTRACT WAVEFORMS (1.5ms refractory period)
    clean_idxs = []
    snippets = []
    pre, post = int(0.001 * sampling_rate), int(0.002 * sampling_rate)
    
    if len(spike_idxs) > 0:
        last_s = -100
        for s in spike_idxs:
            if s - last_s > (0.0015 * sampling_rate):
                # Make sure we don't grab data off the edges
                if pre < s < len(filtered) - post:
                    clean_idxs.append(s)
                    snippets.append(filtered[s-pre : s+post])
                    last_s = s
                    
    # Convert indexes to actual seconds for plotting
    spike_times = np.linspace(t_start, t_end, len(filtered))[clean_idxs]

    # --- PLOTTING ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"DCN Single Unit Analysis - Channel {target_channel}", fontsize=16, fontweight='bold')
    
    # PANEL A: The Timeline (Top spanning both columns)
    ax1 = plt.subplot(2, 1, 1)
    t_sec = np.linspace(t_start, t_end, len(filtered))
    ax1.plot(t_sec, filtered, color='black', linewidth=0.5, alpha=0.8)
    ax1.axvspan(bbn_start, bbn_end, color='cyan', alpha=0.2, label='BBN Stimulus ON')
    ax1.axhline(-thresh, color='red', linestyle='--', label=f'Threshold (-{thresh:.2f} uV)')
    
    if len(spike_times) > 0:
        ax1.plot(spike_times, [-thresh]*len(spike_times), 'ro', markersize=3)
        
    ax1.set_title("Continuous Extracellular Recording")
    ax1.set_ylabel("Voltage (uV)")
    ax1.legend(loc="upper right")
    
    # PANEL B: The Waveforms (Bottom Left)
    ax2 = plt.subplot(2, 2, 3)
    t_ms = np.linspace(-1, 2, pre+post)
    for w in snippets[:50]: # Plot up to 50 individual traces
        ax2.plot(t_ms, w, color='purple', alpha=0.2)
    if snippets: # Plot the bold average line
        ax2.plot(t_ms, np.mean(snippets, axis=0), color='black', linewidth=2, label="Mean Waveform")
        
    ax2.set_title(f"Isolated Single Unit (n={len(snippets)} spikes)")
    ax2.set_xlabel("Time from Peak (ms)")
    ax2.set_ylabel("Voltage (uV)")
    ax2.legend()
    
    # PANEL C: Firing Rate Histogram (Bottom Right)
    ax3 = plt.subplot(2, 2, 4)
    bins = np.arange(t_start, t_end, 0.5) # 500ms bins for a smooth PSTH
    ax3.hist(spike_times, bins=bins, color='gray', edgecolor='black')
    ax3.axvspan(bbn_start, bbn_end, color='cyan', alpha=0.2) # Overlay the sound window
    
    ax3.set_title("Peri-Stimulus Time Histogram (PSTH)")
    ax3.set_xlabel("Time (Seconds)")
    ax3.set_ylabel("Spikes per 500ms")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Keep title from overlapping
    plt.show()

if __name__ == "__main__":
    master_dcn_analysis()