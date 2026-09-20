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
        
        self.trained_weights = {}
        self.trained_biases = {}
        self.is_model_trained = False

    def safe_sigmoid(self, z):
        clipped_z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped_z))
    

    def compute_spectral_flux(self, S):

        # positive flux per frame 

        diff = np.diff(S, axis=1)
        pos_diff = np.maximum(0, diff)
        flux = np.sum(pos_diff, axis=0)
        return np.pad(flux, (1,0), mode = "edge")
    
    def extract_frame_stft_metrics(self, S_frame, freqs):

        # Extracts localized spectral indicators for single frame

        power = S_frame ** 2
        sum_power = np.sum(power) + 1e-8
        centroid = np.sum(freqs[: None] * power, axis=0) / sum_power
        spread = np.sqrt(np.sum(((freqs[:, None] - centroid) ** 2) * power, axis = 0) / sum_power)

        flux = self.compute_spectral_flux

        return np.mean(centroid), np.mean(spread), np.mean(flux)
    


    
    def extract_hht_features_on_slice(self, y_slice):    

        if len(y_slice) < 64:
            return np.zeros(10)
        
        try:
            imfs = self.emd_engine(y_slice)
        except Exception:
            imfs = np.array([y_slice])

        if imfs is None or len(imfs) == 0:
            imfs = np.array([y_slice])

        inst_freqs, inst_amps = [], []
        target_imfs = min(2, len(imfs))

        for i in range(target_imfs):
            imf = imfs[i]
            if np.all(imf == 0):
                continue

            analytic = hilbert(imf)
            amp = np.abs(analytic)[:-1]
            phase = np.unwrap(np.angle(analytic))
            freq = np.clip((np.diff(phase) / (2.0 * np.pi)) * self.target_sr, 0, self.target_sr / 2.0)

            inst_amps.append(amp)
            inst_freqs.append(freq)

        if not inst_freqs:
            return np.zeros(10)

        f1, a1 = inst_freqs[0], inst_amps[0]
        sum_a1 = np.sum(a1) + 1e-8

        t1_centroid = np.sum(f1 * a1) / sum_a1
        t1_spread = np.sqrt(max(0.0), np.sum(((f1 - t1_centroid) ** 2) * a1) / sum_a1)
        t1_flux = float(np.mean(np.diff(f1)**2)) if len(f1) > 1 else 0.0

        if len(inst_freqs) > 1:

            f2 = np.concatenate(inst_freqs)
            a2 = np.concatenate(inst_amps)
            sum_a2 = np.sum(a2) + 1e-8
            t2_centroid = np.sum(f2 * a2) / sum_a2
            t2_spread = np.sqrt(max(0.0, np.sum(((f2 - t2_centroid) ** 2) * a2) / sum_a2))
            t2_flux = float(np.mean(np.diff(f1) ** 2)) if len(f1) > 1 else 0.0

        else:
            t2_centroid, t2_spread, t2_flux = t1_centroid, t1_spread, t1_flux

        itch_gap = abs(t1_centroid - t2_centroid)
        icro_beating = float(np.mean(np.diff(a1, n = 2) ** 2)) if len(a1) > 2 else 0.0
        cr_transient = float(np.mean(librosa.feature.zero_crossing_rate(y=y_slice)))

        ht_vector = np.array([
            t1_centroid, t1_spread, t1_flux,
            t2_centroid, t2_spread, t2_flux,
            pitch_gap, micro_beating, zcr_transient,
            t1_centroid * micro_beating



        ])

        return np.nan_to_num(hht_vector)


    def extract_cascade_features(self, y):
        """Core conditional routing mechanism"""

        # Normalizing signal
        y = y / (np.max(np.abs(y)) + 1e-8)

        n_fft, hop_length = 512, 256
        S = np.abs(librosa.stft(y, n_fft = n_fft, hop_length = hop_length))
        freqs = librosa.fft_frequencies(sr = self.target_sr, n_fft = n_fft)

        mfcc_matrix = librosa.freature.mfcc(S = librosa.power_to_db(S ** 2), sr = self. target_sr, n_mfcc = self.n_mfcc)
        mfcc_vector = np.mean(mfcc_matrix, axis = 1)

        frame_flux = self.compute_spectral_flux(S)
        threshold = np.mean(frame_flux) + (self.flux_threshold_std * np.std(frame_flux))
        transient_mask = frame_flux > threshold

        steady_mask = ~transient_mask
        if np.any(steady_mask):
            stft_centroid, stft_spread, stft_flux = self.extract_stft_features(S[:, steady_mask], freqs)
        else:
            stft_centroid, stft_spread, stft_flux = self.extract_stft_features(S, freqs)

        steady_vector = np.array([stft_centroid, stft_spread, stft_flux])

        transient_frame_indices = np.where(transient_mask)[0]


        if len(transient_frame_indices) > 0:
            hht_vectors = []

            for idx in transient_frame_indices:
                sample_start = idx * hop_length
                sample_end = min(sample_start + n_fft, len(y))
                y_slice = y[sample_start: sample_end]

                hht_vec = self.extract_hht_features_on_slice(y_slice)
                hht_vectors.append(hht_vec)

        else:

            transient_vector = np.zeros(10)


        unified_vector = np.concatenate([mfcc_vector, steady_vector, transient_vector])
        return np.nan_to_num(unified_vector)

    def process_audio(self, audio_path, instrument_threshold = 0.5):

        if not self.is_model_trained:
            return {"status": "ERROR", "message": "Model parameters uninitialized."}
        
        y, _ = librosa.load(audio_path, sr = self.target_sr, mono = True)

        if len(y) == 0:
            return {"status": "ERROR", "message": "Empty audio file."}

        X = self.extract_cascade_features(y)

        detections = []
        for label, w in self.trained_weights.items():
            b = self.trained_biases[label]
            prob = self.safe_sigmoid(np.dot(w, X) + b)

            if prob > instrument_threshold:
                detections.append((label, float(prob)))


        return {
            "status": "SUCCESS",
            "detected_instruments": detections,
            "feature_vector_size": len(X)
            }



                    
    


# https://docs.google.com/document/d/1Hme_wYmL-bNk1LjrTs1aH3JC1CFZlh6bU86rvF9_xOA/edit?tab=t.0 