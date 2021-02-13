import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
import librosa

class FeatureProcessor:
    def __init__(self, data_dir='', output_dir='', acc_only=True, is_eval=False, silent=False):
        self._data_dir = data_dir
        self._output_dir = output_dir
        self._acc_only = acc_only
        self._is_eval = is_eval
        self._fs = 4000
        self._vfss_fs = 60
        self._hop_len_s = 0.1
        self._hop_len = int(self._fs * self._hop_len_s)
        self._frame_res = self._fs / float(self._hop_len)
        self._vfss_res = 1 / float(self._vfss_fs)
        self._nb_frames_1s = int(self._frame_res)
        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)
        self._nb_channels = 3
        self._unique_classes = 2
        self._enc = OneHotEncoder(handle_unknown='ignore')
        self._enc.fit([[0],[1]])
        self._silent = silent

    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _load_signal(self, sigf_path):
        sig_file = np.load(sigf_path)
        if self._acc_only:
            sig_data = sig_file[:,1:4]
        else:
            sig_data = sig_file
        return sig_data

    def _spectrogram(self, signal_input):
        _nb_ch = signal_input.shape[1]
        nb_bins = self._nfft // 2
        _nb_frames = int(np.ceil(signal_input.shape[0] / float(self._hop_len)))
        spectra = np.zeros((_nb_frames, nb_bins, _nb_ch), dtype=complex)
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(signal_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra[:, :, ch_cnt] = stft_ch[1:, :_nb_frames].T
        return spectra, _nb_frames
    
    def _extract_spectrogram_for_file(self, signal_filename):
        signal_in = self._load_signal(os.path.join(self._data_dir, 'signals', signal_filename))
        signal_spec, _nb_frames = self._spectrogram(signal_in)
        np.save(os.path.join(self._output_dir, 'features', '{}.npy'.format(signal_filename.split('.')[0])), signal_spec.reshape(_nb_frames, -1))
        return _nb_frames
    
    def _read_label_file(self, label_filename):
        label_file = {
            'participant': list(), 'file_num': list(), 'swallow_num': list(), 'PAs': list(), 'multiple': list(), 'age': list(), 'sex': list(), 'race': list(), 'volume': list(), 'viscosity': list(), 'utensil': list(), 'start': list(), 'end': list()
        }
        fid = open(os.path.join(self._data_dir, 'labels', label_filename), 'r')
        next(fid)
        for line in fid:
           split_line = line.strip().split(',')
           label_file['participant'].append(split_line[0])
           label_file['file_num'].append(split_line[1])
           label_file['swallow_num'].append(split_line[2])
           label_file['PAs'].append(float(split_line[3]))
           label_file['multiple'].append(split_line[4])
           label_file['age'].append(float(split_line[5]))
           label_file['sex'].append(split_line[6])
           label_file['race'].append(split_line[7])
           label_file['volume'].append(split_line[8])
           label_file['viscosity'].append(split_line[9])
           label_file['utensil'].append(split_line[10])
           label_file['start'].append(int(np.floor(float(split_line[11])*self._vfss_res*self._frame_res)))
           label_file['end'].append(int(np.ceil(float(split_line[12])*self._vfss_res*self._frame_res)))
        fid.close()
        return label_file

    def _get_signal_labels(self, _label_file, _nb_frames):
        swe_label = np.zeros((_nb_frames, 1))
        for i, swe_start in enumerate(_label_file['start']):
            start_frame = swe_start
            end_frame = _nb_frames if _label_file['end'][i] > _nb_frames else _label_file['end'][i]
            swe_label[start_frame:end_frame + 1, :] = 1
        #swe_label = self._enc.transform(swe_label)
        return swe_label#swe_label.toarray()

    def _featlabel_extractor(self):
        Path(os.path.join(self._output_dir, 'features')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self._output_dir, 'labels')).mkdir(parents=True, exist_ok=True)
        df_slices = []
        print('Extracting spectrograms into:{}\n'.format(os.path.join(self._output_dir, 'features')))
        for file_cnt, file_name in enumerate(os.listdir(os.path.join(self._data_dir, 'signals'))):
            if not self._silent:
                print('{}: {}'.format(file_cnt, file_name))
            fname = file_name.split('.')[0]
            _nb_frames = self._extract_spectrogram_for_file('{}.npy'.format(fname))
            label_file = self._read_label_file('{}.csv'.format(fname))
            swe_label = self._get_signal_labels(label_file, _nb_frames)
            np.save(os.path.join(self._output_dir, 'labels', '{}.npy'.format(fname)), swe_label)
            df_slices.append([fname, _nb_frames])
        data_records = pd.DataFrame(df_slices,columns=['fname', 'nb_frames'])
        data_records.to_csv(os.path.join(self._output_dir, 'dataset_frame_record_ws{}_ov{}'.format(self._win_len,self._hop_len) + '.csv'), index=False)
    
    def _features_processor(self):
        wts_file = os.path.join(self._output_dir, 'normalization_wieghts_file')
        print('Estimating weights for normalization:\n\t{}'.format(os.path.join(self._output_dir, 'features')))
        spec_scaler = preprocessing.StandardScaler()
        for file_cnt, file_name in enumerate(os.listdir(os.path.join(self._output_dir, 'features'))):
            if not self._silent:
                print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._output_dir, 'features', file_name))
            spec_scaler.partial_fit(np.concatenate((np.abs(feat_file), np.angle(feat_file)), axis=1))
            del feat_file
        joblib.dump(spec_scaler, wts_file)
        print('Normalized_features_wts_file: {}. Saved.'.format(wts_file))

        print('Normalizing features from:\n\t{}'.format(os.path.join(self._output_dir, 'features')))
        Path(os.path.join(self._output_dir, 'nfeatures')).mkdir(parents=True, exist_ok=True)
        for file_cnt, file_name in enumerate(os.listdir(os.path.join(self._output_dir, 'features'))):
            if not self._silent:
                print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._output_dir, 'features', file_name))
            feat_file = spec_scaler.transform(np.concatenate((np.abs(feat_file), np.angle(feat_file)), axis=1))
            np.save(os.path.join(self._output_dir, 'nfeatures', file_name), feat_file)
            del feat_file
        print('normalized files written to {}'.format(os.path.join(self._output_dir, 'nfeatures')))

class RawFeatureProcessor:
    def __init__(self, data_dir='', output_dir='', acc_only=True, is_eval=False, silent=False):
        self._data_dir = data_dir
        self._output_dir = output_dir
        self._acc_only = acc_only
        self._is_eval = is_eval
        self._fs = 4000
        self._vfss_fs = 60
        self._vfss_res = 1 / float(self._vfss_fs)
        self._hop_len_s = 0.5*self._vfss_res
        self._hop_len = int(self._fs * self._hop_len_s)
        self._frame_res = self._fs / float(self._hop_len)
        self._nb_frames_1s = int(self._frame_res)
        self._win_len = 2 * self._hop_len
        self._nb_channels = 3
        self._unique_classes = 2
        self._enc = OneHotEncoder(handle_unknown='ignore')
        self._enc.fit([[0],[1]])
        self._silent = silent

    def _load_signal(self, sigf_path):
        sig_file = np.load(sigf_path)
        if self._acc_only:
            sig_data = sig_file[:,1:4]
        else:
            sig_data = sig_file
        return sig_data

    def _spatio_splitter(self, signal_input):
        _nb_ch = signal_input.shape[1]
        _nb_frames = int(np.floor((signal_input.shape[0]-self._hop_len) / float(self._hop_len)))
        spatio = np.zeros((_nb_frames, self._win_len, _nb_ch))
        for fr_cnt in range(_nb_frames):
            spatio[fr_cnt, :, :] = signal_input[(fr_cnt*self._hop_len):(fr_cnt*self._hop_len+self._win_len),:]
        return spatio, _nb_frames

    def _extract_spatio_for_file(self, signal_filename):
        signal_in = self._load_signal(os.path.join(self._data_dir, 'signals', signal_filename))
        signal_spatio, _nb_frames = self._spatio_splitter(signal_in)
        np.save(os.path.join(self._output_dir, 'features', '{}.npy'.format(signal_filename.split('.')[0])), signal_spatio.reshape(_nb_frames, -1))
        return _nb_frames

    def _read_label_file(self, label_filename):
        label_file = {
            'participant': list(), 'file_num': list(), 'swallow_num': list(), 'PAs': list(), 'multiple': list(), 'age': list(), 'sex': list(), 'race': list(), 'volume': list(), 'viscosity': list(), 'utensil': list(), 'start': list(), 'end': list()
        }
        fid = open(os.path.join(self._data_dir, 'labels', label_filename), 'r')
        next(fid)
        for line in fid:
           split_line = line.strip().split(',')
           label_file['participant'].append(split_line[0])
           label_file['file_num'].append(split_line[1])
           label_file['swallow_num'].append(split_line[2])
           label_file['PAs'].append(float(split_line[3]))
           label_file['multiple'].append(split_line[4])
           label_file['age'].append(float(split_line[5]))
           label_file['sex'].append(split_line[6])
           label_file['race'].append(split_line[7])
           label_file['volume'].append(split_line[8])
           label_file['viscosity'].append(split_line[9])
           label_file['utensil'].append(split_line[10])
           label_file['start'].append(int(np.floor(float(split_line[11])*self._vfss_res*self._frame_res)))
           label_file['end'].append(int(np.ceil(float(split_line[12])*self._vfss_res*self._frame_res)))
        fid.close()
        return label_file

    def _get_signal_labels(self, _label_file, _nb_frames):
        swe_label = np.zeros((_nb_frames, 1))
        for i, swe_start in enumerate(_label_file['start']):
            start_frame = swe_start
            end_frame = _nb_frames if _label_file['end'][i] > _nb_frames else _label_file['end'][i]
            swe_label[start_frame:end_frame + 1, :] = 1
        #swe_label = self._enc.transform(swe_label)
        return swe_label#swe_label.toarray()

    def _featlabel_extractor(self):
        Path(os.path.join(self._output_dir, 'features')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self._output_dir, 'labels')).mkdir(parents=True, exist_ok=True)
        df_slices = []
        print('Extracting spectrograms into:{}\n'.format(os.path.join(self._output_dir, 'features')))
        for file_cnt, file_name in enumerate(os.listdir(os.path.join(self._data_dir, 'signals'))):
            if not self._silent:
                print('{}: {}'.format(file_cnt, file_name))
            fname = file_name.split('.')[0]
            _nb_frames = self._extract_spatio_for_file('{}.npy'.format(fname))
            label_file = self._read_label_file('{}.csv'.format(fname))
            swe_label = self._get_signal_labels(label_file, _nb_frames)
            np.save(os.path.join(self._output_dir, 'labels', '{}.npy'.format(fname)), swe_label)
            df_slices.append([fname, _nb_frames])
        data_records = pd.DataFrame(df_slices,columns=['fname', 'nb_frames'])
        data_records.to_csv(os.path.join(self._output_dir, 'dataset_frame_record_ws{}_ov{}'.format(self._win_len,self._hop_len) + '.csv'), index=False)
    