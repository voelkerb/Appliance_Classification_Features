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
import features as feat
from numbers import Number
import matplotlib.pyplot as plt

# v and i are numpy arrays of equal length with values in Volt and Ampere
# sr is the samplingrate of the data
f = feat.calculate(v,i,sr)

# Print skalar features
for k in f:
  if isinstance(f[k], Number): 
    print("{}: {}".format(k, round(f[k], 2)))
  else:
    print("{}: {}".format(k, type(f[k])

fig, (ax1, ax2) = plt.subplots(2)
# Plot the avg current and voltage waveform
ax1.title.set_text('Avg. Voltage and Current WF')
ax1.plot(f["I_WF"])
ax1.set_ylabel("Current [A]")
ax1_2 = ax1.twinx()
ax1_2.plot(f["U_WF"])
ax1_2.set_ylabel("Voltage [V]")
ax1.set_xlabel("Sample")

# It's an FFT of size 1024
xf = np.linspace(0.0, 0.5*sr, 1024)
ax2.title.set_text('Frequency Spectrum')
ax2.plot(xf, 2.0/N * np.abs(yf[:N//2]))
ax2.semilogy(xf, np.sqrt(f["FFT"]))
ax2.set_ylabel("Spectrum [RMS]")
ax2.set_xlabel("Frequency [Hz]")

plt.show()
```
