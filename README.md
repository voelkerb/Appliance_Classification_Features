# An Appliance Classification Features Set

This module calculates various features found in litarature for the task of appliance classification in both time and frequency domain.
It requires the raw voltage and current waveforms (in SI units) of at least 300ms length and the samplingrate as input. 
Some features such as LAT, COT, ICR and MIR are only meaningfull if an appliance switch-on event is includes in the first main period of the data.

To calculate all frequency domain features, the samplingrate should be higher than 2kHz (-> Nyquist-Shannon).

The function returns a <em>dictionary</em> containing the following features (with the abbrevation used as dictionary keys):

## Time domain:

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
- V-I Trajectory (VIT)
- Inrush current ratio (ICR)

## Frequency domain:

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
