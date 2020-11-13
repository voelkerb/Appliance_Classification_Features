import numpy as np
import scipy
from scipy import signal
import math
import pywt

def fft(data: np.array, sr:int, win:str=None, scale:str="mag", beta:float=9.0):
    r"""
    Calculate a normalized one sided rfft of the given one dimensional input data.
    Kudos go to Florian Wolling (florian.wolling@uni-siegen.de)

    :param  data: waveform
    :type   data: np.array
    :param    sr: Samplingrate of waveform
    :type     sr: int
    :param   win: Window to apply, one of \"hamming\", \"hanning\", \"blackman\", \"kaiser\".
                  If window unknown, box window is used.
    :type    win: str
    :param scale: Returned fft scaling. One of \"mag\"=magnitude, \"pwr\"=power spectrum
    :type  scale: str
    :param  beta: Beta value of kaiser window
    :type   beta: float

    :return: tuple of fft result and frequencies
    :rtype: tuple
    """
    # Construct window
    win_data = np.ones(len(data))
    if win == "hamming": win_data = np.hamming(len(data))
    elif win == "hanning": win_data = np.hanning(len(data))
    elif win == "blackman": win_data = np.blackman(len(data))
    elif win == "kaiser": win_data = np.kaiser(len(data), beta)
    # Apply windowed FFT
    fft = np.abs(np.fft.rfft(win_data*data))/len(data)
    # Scale
    if scale == "mag": fft = fft*2
    elif scale == "pwr": fft = fft**2
    # Calc freqs
    xf = np.fft.rfftfreq(len(data), 1.0/sr)
    return fft, xf

def goertzel(samples, sample_rate, *freqs):
    """
    Credits: https://stackoverflow.com/a/13512168 
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.

    `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.

    The function returns 2 arrays, one containing the actual frequencies calculated,
    the second the coefficients `(real part, imag part, power)` for each of those frequencies.
    For simple spectral analysis, the power is usually enough.

    Example of usage : 

        # calculating frequencies in ranges [400, 500] and [1000, 1100]
        # of a windowed signal sampled at 44100 Hz

        freqs, results = goertzel(some_samples, 44100, (400, 500), (1000, 1100))
    """
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    freqs = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y  = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2**2 + d1**2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results

def rms(data: np.array):
    """Rerurn RMS value of array"""
    return np.sqrt(data.dot(data)/data.size)

def calculate(v: np.array, i: np.array, sr: int):
    """
    Calculate a set of feature dedicated for the task of appliance classification.

    :param v: Voltage waveform.
    :type  v: np.array
    :param i: Current waveform.
    :type  i: np.array
    :param sr: Samplingrate.
    :type  sr: int

    :return: Features with abbrevation of feature name as key
    :rtype: dict 
    """
    # TODO: Sanity checks before using the sample
    # Max/min current out of bounds? Voltage reasonable
    # Current larger in [0.5s:1.5s] (after event) compared to [0:0.5s] (before event)
    
    # Number of samples for a complete main phase
    sfos = int(math.ceil(sr/50.0))
    
    # Estimate Line Frequency by number of voltage zero crossings
    time = len(v)/sr
    signs = np.sign(v)
    # Since np.sign(0) yields 0, we treat them as negative sign here
    signs[signs == 0] = -1
    zc = len(np.where(np.diff(signs))[0])/2*time
    lineFreq = 60
    if abs(zc-50.0) < abs(zc-60.0): lineFreq = 50
    print(lineFreq)

    # We want a full number of sines
    e = int(int(len(v)/sfos))*sfos

    data = np.recarray((e,), dtype=[("v", np.float32), ("i", np.float32)]).view(np.recarray)
    data["v"] = v[:e]
    data["i"] = i[:e]

    # Features:
    # *** TIME DOMAIN *** **********************************
    # ************ Active power *************************************
    momentary_power = np.array(data["v"]*data["i"])

    # bringing down to 50 Hz by using mean
    #p = np.mean(momentary_power.reshape((-1, sfos)), axis=1)
    p = np.mean(momentary_power)
    
    # ************ Apparent power ***********************************
    # v_ = data["v"].reshape((-1, sfos))
    # c_ = data["i"].reshape((-1, sfos))
    # quicker way to do this
    # vrms = np.sqrt(np.einsum('ij,ij->i', v_, v_)/sfos)
    # irms = np.sqrt(np.einsum('ij,ij->i', c_, c_)/sfos)
    # Because unit of current is in mA
    vrms = rms(data["v"])
    irms = rms(data["i"])
    s = vrms*irms

    # ************ Reactive power ***********************************
    q = np.sqrt(np.abs(s*s - p*p))

    # ************ Resistance ***************************************
    R = vrms/irms
    Y = 1.0/R

    # ********************* Crest Factor***********************************
    CF = max(abs(data["i"]))/rms(data["i"])

    # ********************* Form Factor ***********************************
    # To distinguish switching power supplies from other loads
    FF = rms(data["i"])/ np.mean(np.abs(data["i"]))

    # ********************* Log Attack Time *******************************
    # Time until maxiumum current is drawn. Appliances like power drills
    # with internal speed controll will show larger values here
    # This is only interesting for rush in data
    periods = [data[i*sfos:(i+1)*sfos] for i in range(int(len(data["i"])/sfos))]
    cleanPeriods = periods[1:] # first period has startup peaks
    curOfPeriod = [rms(period["i"]) for period in periods]
    normCurOfPeriod = list(curOfPeriod/max(curOfPeriod))
    # + 1 to accomodate log(0)
    LAT = np.log(normCurOfPeriod.index(max(normCurOfPeriod))*20.0 + 1)

    # ********************* Temporal Centroid *****************************
    # Temporal balancing point of current energy
    TC = 0
    for i, cur in enumerate(curOfPeriod):
        TC += (i+1)*cur
    TC /= sum(curOfPeriod)*sr

    # ********************* Positive Negative Half Cycle Ratio ************
    # Some appliances with e.g. dimmers or speed controllers show different
    # behavior in the positive compared to the negative halfcycle. An
    # Average over multiple halfcycles is taken and the rms of both are
    # compared
    posHal = sum([period["i"][0:int(sfos/2)] for period in cleanPeriods]) / len(cleanPeriods)
    negHal = sum([period["i"][int(sfos/2):] for period in cleanPeriods]) / len(cleanPeriods)
    rmsPosHal = rms(posHal)
    rmsNegHal = rms(negHal)
    if rmsPosHal >= rmsNegHal: PNR = rmsNegHal/rmsPosHal
    else: PNR = rmsPosHal/rmsNegHal

    # ********************* Max-Min Ratio *********************************
    # Alternative way to cover one sided waveform characteristics-
    minI = abs(min(data["i"]))
    maxI = abs(max(data["i"]))
    if maxI >= minI: MAMI = minI/maxI
    else: MAMI = maxI/minI

    # ********************* Peak-Mean Ratio *******************************
    # Determine if appliance has pure sine current of spikes from
    # switching artifacts
    PMR = max(abs(data["i"])) / np.mean(abs(data["i"]))

    # ********************* Max-Inrush Ratio ******************************
    MIR = rms(periods[0]["i"])/max(abs(periods[0]["i"]))

    # ********************* Mean-Variance Ratio ***************************
    # Indicator of the current steadiness. To distinguish e.g. Lightbulbs
    # from pure linear loads (e.g. heater)
    MVR = np.mean(abs(data["i"])) / np.var(abs(data["i"]))
    
    # ********************* Waveform Distortion ***************************
    I_WF = np.mean([cleanPeriods[i]["i"] for i in range(10)], axis=0)
    I_WF_N = I_WF/rms(I_WF)
    x = np.arange(sr/50)
    Y_sin = np.sqrt(2)*np.sin(2 * np.pi * 50 * x / sr)
    WFD = sum(abs(Y_sin) - abs(I_WF_N))

    # ********************* Waveform Approximation ************************
    WFA = np.float32(np.interp(np.arange(0, len(I_WF), len(I_WF)/20), np.arange(0, len(I_WF)), I_WF))
    
    # ********************* Current Over Time ************************
    COT = np.array([rms(periods[i]["i"]) for i in range(25)])

    # ********************* Periode to Steady State ***********************
    L_thres = 1.0/8.0*(np.max(COT) - np.median(COT)) + np.median(COT)
    try: PSS = next(i for i in range(25) if COT[i] < L_thres)
    except: PSS = -1

    # ********************* Phase Angle (CosinePhi) ***********************
    # Phase angle between voltage and current. 
    # vFFT = np.fft.fft(np.hamming(NFFT)*data["v"][:NFFT], NFFT)
    # iFFT = np.fft.fft(np.hamming(NFFT)*data["i"][:NFFT], NFFT)
    # fiftyIndex = int(50 / (sr / NFFT))
    # phi = np.angle(vFFT[fiftyIndex])-np.angle(iFFT[fiftyIndex])
    COS_PHI = p/s


    # ********************* VI Curve ***********************************
    # Normalize to accomodate Voltage fluctuations
    U_WF = np.mean([cleanPeriods[i]["v"] for i in range(10)], axis=0)
    U_norm = U_WF/max(-1*min(U_WF), max(U_WF))
    I_norm = I_WF/max(-1*min(I_WF), max(I_WF))
    # See if samples for one sine is integer multiple of 20
    # If not upsample it to next
    if sfos/20.0 - int(sfos/20.0) > 0:
        n = int((int(sfos/20.0) + 1)*20)
        U_norm2 = signal.resample(U_norm, n)
        I_norm2 = signal.resample(I_norm, n)
        VIT = np.array(list(U_norm2[::int(len(U_norm2)/20)]) + list(I_norm2[::int(len(I_norm2)/20)]))
    else:
        VIT = np.array(list(U_norm[::int(sfos/20)][:20]) + list(I_norm[::int(sfos/20)][:20]))


    # ********************* Inrush Current Rashio ***********************
    ICR = rms(periods[0]["i"])/rms(periods[-1]["i"])

    # *** FREQUENCY DOMAIN *** **********************************
    # NFFT = 1024
    NFFT = len(data)
    FFT, f = fft(data["i"], sr, win="hamming")

    # NFFT_2 = math.floor(NFFT/2)
    # Use cleaned FFT
    #f, FFT = scipy.signal.welch(data["i"], sr, nperseg=NFFT, scaling="spectrum", window="hamming", average='median')
    # Use FFT
    # hamData = np.hamming(NFFT)*data["i"][:NFFT]
    # FFT = np.abs(np.fft.rfft(hamData, norm="ortho", n=NFFT)[:NFFT//2]/NFFT)
    # f = [i * (sr / NFFT) for i in range(int(NFFT_2))]

    # ********************* Harmonic Energy Distribution ******************
    # Vector of 20 values; Magnitudes of 50, 100, 150, ..., 1000Hz
    # Calculate Harmonic Energy Distribution
    HEDFreqs = [float(i*lineFreq) for i in range(1, 20)]
    
    # goertz = [goertzel(data["i"], sr, [f-2, f+2]) for f in HEDFreqs]
    # print(goertz[0])

    # Use margin around goal freq to calculate single harmonic magnitude
    # NOTE: This is not nice
    indices = [[i for i, f_ in enumerate(f) if abs(f_-freq_) < 3] for freq_ in HEDFreqs]
    #for fr, ind in zip(HEDFreqs, indices):
    #    print("{}: {} - {} - {}".format(fr, ind, f[ind], spec[ind]))
    Harm = [sum(FFT[ind]) if len(ind) > 0 else 0 for ind in indices]
    if Harm[0] == 0: raise AssertionError("Coul not determine Harmonics, maybe increase FFT size")
    HED = Harm[1:]/Harm[0]
    
    # ********************* Total Harmonic Distortion *********************
    THD = 10*np.log10(sum(Harm[1:6])/Harm[0])

    # ********************* Spectral Flatness ****************************
    # Measure for energy distribution in freq spectrum.
    # A Value of 1.0 is equivalent to white noise. The closer to 0, the
    # stronger are individual frequencies (linear loads -> low values)
    NormFFT = FFT/max(FFT)
    SPF = (scipy.stats.mstats.gmean(NormFFT))/(sum(NormFFT)/len(NormFFT))


    # ********************* Odd-Even Harmonics Ratio **********************
    # Some (most) appliances show an imbalanced odd (150, 250, 350Hz ...)
    # to even (100, 200, 300Hz, ...) rate
    OER = np.mean(HEDFreqs[1::2])/np.mean(HEDFreqs[2::2])
    
    # ********************* Tristimulus ***********************************
    # Energy of different harmonic groups (low, medium, high - harmonics)
    TRI = np.array([Harm[1]/sum(Harm), sum(Harm[2:5])/sum(Harm), sum(Harm[5:11])/sum(Harm)])

    # ********************* Spectral Centroid *****************************
    # Same as temporal centroid but for the frequency spectrum
    SC = sum([xf*(bin*(sr/NFFT)) for bin, xf in enumerate(FFT)]) / sum(FFT)
    
    # ***************** Wavelet transform **************************
    # Needs to be investigated further. Waveform has better FREQUENCY
    # resolution for lower frequencies
    cA, cD = pywt.dwt(data["i"], 'db1')
    WT = [cA, cD]
    
    x = {
        "LF":       lineFreq, 
        "P":        p, 
        "Q":        q, 
        "S":        s, 
        "R":        R, 
        "Y":        Y, 
        "CF":       CF, 
        "FF":       FF, 
        "LAT":      LAT, 
        "TC":       TC, 
        "PNR":      PNR,
        "MAMI":     MAMI, 
        "PMR":      PMR, 
        "MIR":      MIR, 
        "MVR":      MVR, 
        "WFD":      WFD, 
        "WFA":      WFA, 
        "COT":      COT, 
        "PSS":      PSS, 
        "COS_PHI":  COS_PHI, 
        "VIT":      VIT, 
        "ICR":      ICR, 
        "HED":      HED, 
        "THD":      THD, 
        "SPF":      SPF, 
        "OER":      OER, 
        "TRI":      TRI, 
        "SC":       SC, 
        "WT":      WT,
    }
    # Additional stuff
    x["I_WF"] = I_WF
    x["U_WF"] = U_WF
    x["FFT"] = FFT

    return x