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

        fig, (ax) = plt.subplots(1)
        # Plot the avg current and voltage waveform
        ax.title.set_text('Input data')
        ax.plot(data["i"])
        ax.set_ylabel("Current [mA]")
        ax.set_xlabel("Sample")
        ax.axvline(x=int(0.5*sr), linewidth=2, color=(0,0,0))

        fig, (ax1, ax2) = plt.subplots(2)
        # Plot the avg current and voltage waveform
        ax1.title.set_text('Avg. Voltage and Current WF')
        ax1_2 = ax1.twinx()
        ax1_2.plot(f["U_WF"], c='orange', label="Voltage")
        ax1.plot(f["I_WF"], label="Current")
        ax1.set_ylabel("Current [mA]")
        ax1_2.set_ylabel("Voltage [V]")
        ax1.set_xlabel("Sample")
        plt.legend()

        # It's an FFT of size 1024
        xf = np.linspace(0.0, 0.5*sr, 512)
        ax2.title.set_text('Frequency Spectrum')
        ax2.semilogy(xf, np.sqrt(f["FFT"][:512]))
        ax2.set_ylabel("Spectrum [RMS]")
        ax2.set_xlabel("Frequency [Hz]")

        plt.show()
# %%
