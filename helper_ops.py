import numpy as np
from scipy import signal
import scipy.signal as sp_sig
import scipy.io.wavfile

def rms (x):
    # compute rms of a matrix
    sq = np.mean(np.square(x), axis = 0)    
    return np.sqrt(sq)

def slice_1dsignal(signal, window_size, winshift, minlength, left_context=256, right_context=256):
    """ 
    Return windows of the given signal by sweeping in stride fractions
    of window
    Slices that are less than minlength are omitted
    """
    # concatenate zeros to beginning for adding context
    n_samples = signal.shape[0]
    num_slices = (n_samples)
    slices = [] # initialize empty array 

    for beg_i in range(0, n_samples, winshift):
        beg_i_context = beg_i - left_context
        end_i = beg_i + window_size + right_context
        if n_samples - beg_i < minlength :
            break
        if beg_i_context < 0 :
            slice_ = np.concatenate((np.zeros((1, left_context - beg_i)),np.array([signal[:end_i]])), axis=1)
        elif end_i <= n_samples :
            slice_ = np.array([signal[beg_i_context:end_i]])
        else :
            slice_ = np.concatenate((np.array([signal[beg_i_context:]]), np.zeros((1, end_i - n_samples))), axis=1)
        # print(slice_.shape)
        slices.append(slice_)
    slices = np.vstack(slices)
    slices = np.expand_dims(slices, axis=2) # the CNN will need 3D data
    return slices

def QERB_calculation(bmm,cfs,fs):
    central = cfs.shape[0]
    samples = bmm.shape[1]
    half = samples/2
    F = np.zeros((samples , central))
    G = np.zeros((samples , central))
    max_val = np.zeros(central)
    ener = np.zeros(central)
    BW = np.zeros(central)
    QdB = np.zeros(central)
    
    for i in range(int(central)):
        F[:,i] = (2*abs(np.fft.fft(bmm[i,:]))/samples)**2
        max_val[i] = F.max(0)[i]
        for j in range(int(half)+1):
            ener[i] = ener[i]+F[j,i]
        #ener[i] = (F.sum(0)[i])/2
        BW[i] = (ener[i]/max_val[i]) * fs/samples
        QdB[i] = cfs[i]/BW[i]
    return QdB

def get_dpoae(tl_bmm, cf_location=0,sig_start=0):
    # get the fft of last channel to predict the dpoae
    oae_sig = tl_bmm[0, sig_start: ,cf_location] # pick a CF
    oae_fft = np.fft.fft(oae_sig)
    nfft = oae_fft.shape[0]
    return np.absolute(oae_fft[:int(nfft/2)]), nfft


def concatenate_tl_pred (tl_pr):
    tl_2d = []
    for i in range(tl_pr.shape[0]):
        tl_2d.append(tl_pr[i])
    return np.expand_dims(np.vstack(tl_2d), axis=0)

def undo_window(tl_pr, winlength, winshift, ignore_first_set=0, fs = 20e3):
    trailing_silence = 0.
    nframes = tl_pr.shape[0]
    slength = ((nframes - 1)) * winshift + winlength
    tl_2d = np.zeros((slength, tl_pr.shape[2]))
    scale_ = np.zeros((slength,1))
    dummyones = np.ones((tl_pr.shape[0], tl_pr.shape[1]))
    trailing_zeros = int(trailing_silence * fs)
    sigrange = range (winlength)
    tl_2d [sigrange, :] = tl_2d [sigrange, :] + tl_pr[0]
    scale_[sigrange,0] = scale_[sigrange,0] + dummyones[0]
    for i in range(1,nframes):
        sigrange = range (i * winshift + ignore_first_set, (i*winshift) + winlength)
        tl_2d [sigrange, :] = tl_2d [sigrange, :] + tl_pr[i,ignore_first_set:,:]
        scale_[sigrange,0] = scale_[sigrange,0] + dummyones[i,ignore_first_set:]
    
    tl_2d /= scale_
    return np.expand_dims(tl_2d[trailing_zeros:,:], axis=0)

def wavfile_read(wavfile,fs=[]):
    # if fs is given the signal is resampled to the given sampling frequency
    fs_signal, speech = scipy.io.wavfile.read(wavfile)
    if not fs:
        fs=fs_signal

    if speech.dtype == 'int16':
        nb_bits = 16 # -> 16-bit wav files
    elif speech.dtype == 'int32':
        nb_bits = 32 # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    speech = speech / (max_nb_bit + 1.0) # scale the signal to [-1.0,1.0]

    if fs_signal != fs :
        signalr = sp_sig.resample_poly(speech, fs, fs_signal)
    else:
        signalr = speech

    return signalr, fs

