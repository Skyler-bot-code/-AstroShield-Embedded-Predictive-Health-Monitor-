import os
import json
import warnings
import numpy as np
import librosa
from scipy.signal import medfilt, hilbert
from PyEMD import EMD

class TransientGatedH2MB():

    def __init__(self, target_sr= 160000, n_mfcc = 13, flux_threshold_std = 1.5, local_window_frames = 125):
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.flux_threshold_std = flux_threshold_std
        self.local_window_frames = local_window_frames
        self.emd_engine = EMD()



        self.feature_dim = self.n_mfcc + 13
        self.is_model_trained = False

    def compute_frame_flux(self, S):

        # positive flux per frame 

        diff = np.diff(S, axis=1)
        pos_diff = np.maximum(0, diff)
        flux = np.sum(pos_diff, axis=0)
        return np.pad(flux, (1,0), mode = "edge")
    
    def extract_frame_stft_metrics(self, S_frame, freqs):

        # Extracts localized spectral indicators for single frame

        power = S_frame ** 2
        sum_power = np.sum(power) + 1e-8
        centroid = np.sum(freqs * power) / sum_power
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / sum_power)
        return centroid, spread
    
    def extract_hht_features_on_slice(self, y_slice):
        # Targeted EMD/Hilbert demodulation on a raw window slice.
        
        window = np.hanning(len(y_slice))
        y_windowed = y_slice * window

        if len(y_windowed) < 64:
            return np.zeros(10)
        
        try:
            imfs = self.emd_engine(y_windowed)
        except Exception:
            imfs = np.array([y_windowed])

        if imfs is None or len(imfs) == 0:
            imfs = np.array([y_windowed])

        isnt_freqs, inst_amps = [], []




        target_imfs = min(2, len(imfs))

        for i in range(target_imfs):
            ...
            
    
    def extract_sequence_features(self, y):
        # Processes frame by frame audio chronologically
        # Return 2D array
        ...

# https://docs.google.com/document/d/1Hme_wYmL-bNk1LjrTs1aH3JC1CFZlh6bU86rvF9_xOA/edit?tab=t.0 