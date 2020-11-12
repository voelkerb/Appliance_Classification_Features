# %%
import features.features as feat
from numbers import Number
import matplotlib.pyplot as plt
import pickle
import numpy as np


with open("sample.data",'rb') as f: 
    Data = pickle.load(f)
    sr = 2000 

    for data, device in Data:
        print(device + ":")
        # Appliance turned on after 0.5s 
        v = data["v"][int(0.5*sr):]
        i = data["i"][int(0.5*sr):]*0.001 # was in mA
        # v and i are numpy arrays of equal length with values in Volt and Ampere
        # sr is the samplingrate of the data
        f = feat.calculate(v,i,sr)

        # Print features
        for k in f:
            if isinstance(f[k], Number): 
                print(" {:>8}:  {:7.2f}".format(k, round(float(f[k]), 2)))
            else:
                print(" {:>8}: [{:7.2f} ... {:7.2f}]".format(k, f[k][0], f[k][-1]))

        fig, (ax0, ax1, ax3) = plt.subplots(3, figsize=(8,8), tight_layout=True)
        # Plot the input data
        ax0.title.set_text('Input data')
        ax0.plot(data["i"]*0.001)
        ax0.set_ylabel("Current [A]")
        ax0.set_xlabel("Sample")
        ax0.axvline(x=int(0.5*sr), linewidth=2, color=(0,0,0))

        # Plot the avg current and voltage waveform
        ax1.title.set_text('Avg. Voltage and Current WF')
        ax2 = ax1.twinx()
        lns1 = ax1.plot(f["I_WF"], label="Current")
        lns2 = ax2.plot(f["U_WF"], c='orange', label="Voltage")
        # Plot formatting
        ax1.set_ylabel("Current [mA]")
        ax2.set_ylabel("Voltage [V]")
        ax1.set_xlabel("Sample")
        ax1.legend(lns1+lns2, [l.get_label() for l in lns1+lns2], loc=0)
        
        # Plot the Frequency Spectrum
        xf = np.linspace(0.0, 0.5*sr, 512) # FFT size is 1024
        ax3.title.set_text('Frequency Spectrum')
        ax3.semilogy(xf, np.sqrt(f["FFT"][:512]))
        ax3.set_ylabel("Spectrum [RMS]")
        ax3.set_xlabel("Frequency [Hz]")

        plt.show()
# %%
