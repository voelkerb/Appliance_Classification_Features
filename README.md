# An Appliance Classification Features Set

This module calculates various features found in litarature for the task of appliance classification in both time and frequency domain.
It requires the raw voltage and current waveforms (in SI units) of at least 300ms length and the samplingrate as input. 
Some features such as LAT, COT, ICR and MIR are only meaningfull if an appliance switch-on event is includes in the first main period of the data.

To calculate all frequency domain features, the samplingrate should be higher than 2kHz (-> Nyquist-Shannon).

The function returns a <em>dictionary</em> containing the following features (with the abbrevation used as dictionary keys):

## Time Domain:

- Active (P), Reactive (Q), Apparent (S) power
- Resistance (R), Admittance (Y)
- Crest factor (CF)
- Form factor (FF)
- Log attack time (LAT)
- Temporal centroid (TC)
- Positive negative half cycle ratio (PNR)
- Max-min ratio (MAMI)
- Peak-mean ratio (PMR)
- Max inrush ratio (MIR)
- Mean-variance ratio (MVR)
- Waveform distortion (WFD)
- Waveform approximation (WFA)
- Current over time (COT)
- Periode to steady state (PSS) 
- Phase angle (COS_PHI)
- V-I trajectory (VIT)
- Inrush current ratio (ICR)

## Frequency Domain:

- Harmonic energy distribution (HED)
- Total harmonic distortion (THD)
- Spectral flatness (SPF)
- Odd-even harmonic ratio (OER)
- Tristimulus (TRI)
- Spectral centroid (SC)
- Wavelet transform (WT)

## Additional:

- 1024 point Fast Fourier Transform (FFT)
- Mean current, voltage waveform (I_WF, U_WF)

## How To Use:
```python
import features.features as feat
from numbers import Number
import matplotlib.pyplot as plt
import numpy as np

# Load data
...

# v and i are numpy arrays of equal length with values in Volt and Ampere
# sr is the samplingrate of the data
f = feat.calculate(v,i,sr)

# Print features
for k in f:
    if isinstance(f[k], Number): 
        print(" {:>8}:  {:7.2f}".format(k, round(float(f[k]), 2)))
    else:
        print(" {:>8}: [{:7.2f} ... {:7.2f}]".format(k, f[k][0], f[k][-1]))

fig, (ax0, ax1, ax3) = plt.subplots(3)
# Plot the input data
ax0.title.set_text('Input data')
ax0.plot(i)
ax0.set_ylabel("Current [A]")
ax0.set_xlabel("Sample")

# Plot the avg current and voltage waveform
lns1 = ax1.plot(f["I_WF"], label="Current")
ax2 = ax1.twinx()
lns2 = ax2.plot(f["U_WF"], c='orange', label="Voltage")
# Format plot
ax1.title.set_text("Avg. Voltage and Current WF of {}".format(device))
ax1.set_ylabel("Current [A]")
ax2.set_ylabel("Voltage [V]")
ax1.set_xlabel("Sample")
ax1.legend(lns1+lns2, [l.get_label() for l in lns1+lns2], loc=0)

# Plot FFT
xf = np.linspace(0.0, 0.5*sr, 512) # FFT size is 1024
ax3.semilogy(xf, np.sqrt(f["FFT"][:512]))
ax3.title.set_text("Frequency Spectrum of {}".format(device))
ax3.set_ylabel("Spectrum [RMS]")
ax3.set_xlabel("Frequency [Hz]")

plt.show()
```

Which gives you a plot like:

<img src="docu\plot.png" width="500">
