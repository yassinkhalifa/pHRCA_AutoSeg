import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from IPython import embed
from collections import deque
import random
from keras.utils import Sequence
from keras.utils.io_utils import HDF5Matrix
import h5py
from sklearn.metrics import confusion_matrix

class data_splitter:
    def __init__(self, data_dir='', win_len=0, hop_len=0, seq_len=0, seq_hop_len=0, nb_channels=3, nb_classes=2, validate=False, nb_folds=10, silent=False):
        self._data_dir = data_dir
        self._win_len = win_len
        self._hop_len = hop_len
        self._validate = validate
        self._nb_folds = nb_folds
        self._seq_len = seq_len
        self._seq_hop_len = seq_hop_len
        self._nb_channels = nb_channels
        self._nb_classes = nb_classes
        self._train_perc = 1-(1/self._nb_folds)
        self._nfft = self._next_greater_power_of_2(self._win_len)
        self._silent = silent

    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()
    
    def perform(self):
        files_list = pd.read_csv(os.path.join(self._data_dir, 'dataset_frame_record_ws{}_ov{}'.format(self._win_len,self._hop_len) + '.csv'))
        files_list = files_list.sample(frac=1).reset_index(drop=True)
        _nb_files = files_list.shape[0]
        _nb_tot_frames = np.sum(files_list.nb_frames)
        _nb_test_frames = int(float(1-self._train_perc)*_nb_tot_frames)
        residual_records = files_list
        Path(os.path.join(self._data_dir, 'folds_metadata')).mkdir(parents=True, exist_ok=True)
        for i_fold in range(self._nb_folds):
            print('Fetching splits in fold# {}: {}'.format(i_fold+1, self._nb_folds))
            test_records = pd.DataFrame(columns=['fname', 'nb_frames'])
            _nb_frames_iter = 0
            while _nb_frames_iter<_nb_test_frames:
                if (not residual_records.empty):
                    sel_file = residual_records.sample(n=1, replace=False)
                    residual_records = residual_records.drop(list(sel_file.index)).reset_index(drop=True)
                    sel_file = sel_file.reset_index(drop=True)
                    _nb_frames_iter += sel_file.nb_frames[0]
                    test_records = test_records.append(sel_file, ignore_index=True)
                else:
                    break
            if ~test_records.empty:
                _nb_val_frames = int(0.5*_nb_test_frames)
                residual_test_records = test_records
                val_records = pd.DataFrame(columns=['fname', 'nb_frames'])
                _nb_val_frames_iter = 0
                while _nb_val_frames_iter<_nb_val_frames:
                    sel_file = residual_test_records.sample(n=1, replace=False)
                    residual_test_records = residual_test_records.drop(list(sel_file.index)).reset_index(drop=True)
                    sel_file = sel_file.reset_index(drop=True)
                    _nb_val_frames_iter += sel_file.nb_frames[0]
                    val_records = val_records.append(sel_file, ignore_index=True)
                train_records = files_list[~files_list.fname.isin(test_records.fname)].reset_index(drop=True)
                train_records = self.get_nb_sequences(train_records)
                train_records = train_records[train_records.nb_seq>1].reset_index(drop=True)
                train_records.to_csv(os.path.join(self._data_dir, 'folds_metadata', 'train_metadata_{}.csv'.format(i_fold+1)), index=False)
                val_records = self.get_nb_sequences(val_records)
                val_records = val_records[val_records.nb_seq>1].reset_index(drop=True)
                val_records.to_csv(os.path.join(self._data_dir, 'folds_metadata', 'val_metadata_{}.csv'.format(i_fold+1)), index=False)
                test_records = residual_test_records
                test_records = self.get_nb_sequences(test_records)
                test_records = test_records[test_records.nb_seq>1].reset_index(drop=True)
                test_records.to_csv(os.path.join(self._data_dir, 'folds_metadata', 'test_metadata_{}.csv'.format(i_fold+1)), index=False)

    def prepare(self, fold_n=0, run_type='train'):
        data_records = pd.read_csv(os.path.join(self._data_dir, 'folds_metadata', '{}_metadata_{}.csv'.format(run_type, fold_n+1)))
        #data_records = self.get_nb_sequences(data_records)
        #data_records = data_records[data_records.nb_seq>1].reset_index(drop=True)
        tot_nb_seq = int(np.sum(data_records.nb_seq))
        seq_data = np.zeros((tot_nb_seq, self._seq_len, int(self._nfft/2), 2*self._nb_channels))
        seq_labels = np.zeros((tot_nb_seq, self._seq_len, 1))#np.zeros((tot_nb_seq, self._seq_len, self._nb_classes))
        seq_cnt = 0
        Path(os.path.join(self._data_dir, 'folds_h5')).mkdir(parents=True, exist_ok=True)
        for file_cnt, file_name in enumerate(data_records.fname):
            if not self._silent:
                print('Fetching sequences in file# {}: {}'.format(file_cnt, file_name))
            file_data = np.load(os.path.join(self._data_dir, 'nfeatures', '{}.npy'.format(file_name)))
            file_labels = np.load(os.path.join(self._data_dir, 'labels', '{}.npy'.format(file_name)))
            for file_seq_cnt in range(data_records.nb_seq[file_cnt]):
                seq_data[seq_cnt,:,:,:] = file_data[(file_seq_cnt*self._seq_hop_len):(file_seq_cnt*self._seq_hop_len+self._seq_len),:].reshape((self._seq_len, int(self._nfft/2), 2*self._nb_channels))
                seq_labels[seq_cnt,:,:] = file_labels[(file_seq_cnt*self._seq_hop_len):(file_seq_cnt*self._seq_hop_len+self._seq_len),:]
                seq_cnt = seq_cnt+1
        h5_file = h5py.File(os.path.join(self._data_dir, 'folds_h5', '{}_metadata_{}.hdf5'.format(run_type, fold_n+1)), 'w')
        h5_file.create_dataset('seq_data', data=seq_data)
        h5_file.create_dataset('seq_labels', data=seq_labels)
        h5_file.close()

    def rawprepare(self, fold_n=0, run_type='train'):
        data_records = pd.read_csv(os.path.join(self._data_dir, 'folds_metadata', '{}_metadata_{}.csv'.format(run_type, fold_n+1)))
        tot_nb_seq = int(np.sum(data_records.nb_seq))
        seq_data = np.zeros((tot_nb_seq, self._seq_len, self._win_len, self._nb_channels))
        seq_labels = np.zeros((tot_nb_seq, self._seq_len, 1))#np.zeros((tot_nb_seq, self._seq_len, self._nb_classes))
        seq_cnt = 0
        Path(os.path.join(self._data_dir, 'folds_h5')).mkdir(parents=True, exist_ok=True)
        for file_cnt, file_name in enumerate(data_records.fname):
            if not self._silent:
                print('Fetching sequences in file# {}: {}'.format(file_cnt, file_name))
            file_data = np.load(os.path.join(self._data_dir, 'features', '{}.npy'.format(file_name)))
            file_labels = np.load(os.path.join(self._data_dir, 'labels', '{}.npy'.format(file_name)))
            for file_seq_cnt in range(data_records.nb_seq[file_cnt]):
                seq_data[seq_cnt,:,:,:] = file_data[(file_seq_cnt*self._seq_hop_len):(file_seq_cnt*self._seq_hop_len+self._seq_len),:].reshape((self._seq_len, self._win_len, self._nb_channels))
                seq_labels[seq_cnt,:,:] = file_labels[(file_seq_cnt*self._seq_hop_len):(file_seq_cnt*self._seq_hop_len+self._seq_len),:]
                seq_cnt = seq_cnt+1
        h5_file = h5py.File(os.path.join(self._data_dir, 'folds_h5', '{}_metadata_{}.hdf5'.format(run_type, fold_n+1)), 'w')
        h5_file.create_dataset('seq_data', data=seq_data)
        h5_file.create_dataset('seq_labels', data=seq_labels)
        h5_file.close()

    def noseq_prepare(self, fold_n=0, run_type='train'):
        data_records = pd.read_csv(os.path.join(self._data_dir, 'folds_metadata', '{}_metadata_{}.csv'.format(run_type, fold_n+1)))
        tot_nb_frames = int(np.sum(data_records.nb_frames))
        frame_data = np.zeros((tot_nb_frames, int(self._nfft/2), 2*self._nb_channels))
        frame_labels = np.zeros((tot_nb_frames, 1))
        frame_cnt = 0
        Path(os.path.join(self._data_dir, 'folds_h5')).mkdir(parents=True, exist_ok=True)
        for file_cnt, file_name in enumerate(data_records.fname):
            if not self._silent:
                print('Fetching sequences in file# {}: {}'.format(file_cnt, file_name))
            file_data = np.load(os.path.join(self._data_dir, 'nfeatures', '{}.npy'.format(file_name)))
            file_labels = np.load(os.path.join(self._data_dir, 'labels', '{}.npy'.format(file_name)))
            frame_data[frame_cnt:(frame_cnt+data_records.nb_frames[file_cnt]),:,:] = file_data[:,:].reshape((data_records.nb_frames[file_cnt], int(self._nfft/2), 2*self._nb_channels))
            frame_labels[frame_cnt:(frame_cnt+data_records.nb_frames[file_cnt]),:] = file_labels[:,:]
            frame_cnt = frame_cnt+data_records.nb_frames[file_cnt]
        h5_file = h5py.File(os.path.join(self._data_dir, 'folds_h5', '{}_metadata_{}.hdf5'.format(run_type, fold_n+1)), 'w')
        h5_file.create_dataset('frame_data', data=frame_data)
        h5_file.create_dataset('frame_labels', data=frame_labels)
        h5_file.close()
            
    def get_nb_sequences(self, data_records):
        nb_seq = np.zeros(data_records.shape[0])
        for file_cnt, file_name in enumerate(data_records.fname):
            if not self._silent:
                print('Counting sequences in file# {}: {}'.format(file_cnt, file_name))
            seq_labels = np.load(os.path.join(self._data_dir, 'labels', '{}.npy'.format(file_name)))
            #nb_seq[file_cnt] = int(np.floor((seq_labels.shape[0]-self._seq_hop_len)/(self._seq_len-self._seq_hop_len)))
            nb_seq[file_cnt] = int(np.floor(seq_labels.shape[0]/self._seq_hop_len)-1)
        data_records['nb_seq'] = np.asarray(nb_seq, dtype=np.int)
        return data_records




class DataGenerator(Sequence):
    def __init__(self, data_dir='', shuffle=False, win_len=800, hop_len=400, seq_len=10, seq_hop_len=5, nb_channels=3, nb_classes=2, n_fold=0, batch_size=16, run_type='train'):
        self._data_dir = data_dir
        self._shuffle = shuffle
        self._win_len = win_len
        self._hop_len = hop_len
        self._nb_channels = nb_channels
        self._nb_classes = nb_classes
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._seq_hop_len = seq_hop_len
        self._n_fold = n_fold
        self._nfft = self._next_greater_power_of_2(self._win_len)
        self._run_type = run_type
        self._nb_total_batches = self._get_nb_total_batches()
        self.on_epoch_end()
        self._data_file = h5py.File(os.path.join(self._data_dir, 'folds_h5', '{}_metadata_{}.hdf5'.format(self._run_type, self._n_fold+1)), 'r')

    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()
    

    def _get_nb_total_batches(self):
        h5_data = HDF5Matrix(os.path.join(self._data_dir, 'folds_h5', '{}_metadata_{}.hdf5'.format(self._run_type, self._n_fold+1)), 'seq_labels')
        return int(np.floor(h5_data.shape[0]/self._batch_size))

    def on_epoch_end(self):
        if self._shuffle:
            return

    def __data_generation(self, index):
        #batch_x = np.transpose(HDF5Matrix(os.path.join(self._data_dir, 'folds_h5', '{}_metadata_{}.hdf5'.format(self._run_type, self._n_fold+1)), 'seq_data',start=int(index*self._batch_size), end=int((index+1)*self._batch_size)), (0, 3, 1, 2))
        #batch_y = HDF5Matrix(os.path.join(self._data_dir, 'folds_h5', '{}_metadata_{}.hdf5'.format(self._run_type, self._n_fold+1)), 'seq_labels',start=int(index*self._batch_size), end=int((index+1)*self._batch_size))
        batch_x = np.transpose(self._data_file['seq_data'][int(index*self._batch_size):int((index+1)*self._batch_size),:,:,:], (0, 3, 1, 2))
        batch_y = self._data_file['seq_labels'][int(index*self._batch_size):int((index+1)*self._batch_size),:,:]
        return batch_x, batch_y
        
    def __len__(self):
        return self._nb_total_batches

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        return X, y


class DataGenerator_noseq(Sequence):
    def __init__(self, data_dir='', shuffle=False, win_len=800, hop_len=400, nb_channels=3, nb_classes=2, n_fold=0, batch_size=64, run_type='train'):
        self._data_dir = data_dir
        self._shuffle = shuffle
        self._win_len = win_len
        self._hop_len = hop_len
        self._nb_channels = nb_channels
        self._nb_classes = nb_classes
        self._batch_size = batch_size
        self._n_fold = n_fold
        self._nfft = self._next_greater_power_of_2(self._win_len)
        self._run_type = run_type
        self._nb_total_batches = self._get_nb_total_batches()
        self.on_epoch_end()
        self._data_file = h5py.File(os.path.join(self._data_dir, 'folds_h5', '{}_metadata_{}.hdf5'.format(self._run_type, self._n_fold+1)), 'r')

    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()
    

    def _get_nb_total_batches(self):
        h5_data = HDF5Matrix(os.path.join(self._data_dir, 'folds_h5', '{}_metadata_{}.hdf5'.format(self._run_type, self._n_fold+1)), 'frame_labels')
        return int(np.floor(h5_data.shape[0]/self._batch_size))

    def on_epoch_end(self):
        if self._shuffle:
            return

    def __data_generation(self, index):
        batch_x = np.transpose(np.stack((np.split(self._data_file['frame_data'][int(index*self._batch_size):int((index+1)*self._batch_size),:,:], 2, axis=-1)), axis=-1), (0, 3, 1, 2))
        batch_y = self._data_file['frame_labels'][int(index*self._batch_size):int((index+1)*self._batch_size),:]
        return batch_x, batch_y
        
    def __len__(self):
        return self._nb_total_batches

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        return X, y


class evaluation_metrics:
    def __init__(self, results_dir='', win_len=800, hop_len=400, seq_len=10, seq_hop_len=5, nb_channels=3, nb_classes=2, silent=False):
        self._results_dir = results_dir
        self._win_len = win_len
        self._hop_len = hop_len
        self._nb_channels = nb_channels
        self._nb_classes = nb_classes
        self._seq_len = seq_len
        self._seq_hop_len = seq_hop_len
        self._silent = silent

    def combine_seqlogits(self, seq_logits, seq_files, run_type='test', fold_n=0):
        print('Combining {} sequences:'.format(run_type))
        seq_cnt = 0
        for file_cnt, file_name in enumerate(seq_files.fname):
            if (seq_cnt+seq_files.nb_seq[file_cnt])>seq_logits.shape[0]:
                break
            if not self._silent:
                print('Combining sequences of file# {}: {}'.format(file_cnt, file_name))
            pred_labels = np.zeros((seq_files.nb_seq[file_cnt],seq_files.nb_frames[file_cnt]))
            seq_labels = seq_logits[seq_cnt:(seq_cnt+seq_files.nb_seq[file_cnt]),:,:]#np.argmax(seq_logits[seq_cnt:(seq_cnt+seq_files.nb_seq[file_cnt]),:,:], axis=-1)
            for file_seq_cnt in range(seq_files.nb_seq[file_cnt]):
                pred_labels[file_seq_cnt,(file_seq_cnt*self._seq_hop_len):(file_seq_cnt*self._seq_hop_len+self._seq_len)] = np.squeeze(seq_labels[file_seq_cnt,:])
            seq_cnt = seq_cnt+seq_files.nb_seq[file_cnt]
            pred_labels = np.any(pred_labels, axis=0)
            pred_labels = pred_labels.astype(int)
            np.save(os.path.join(self._results_dir, '{}_val_fold{:0>2d}'.format(run_type, fold_n+1), '{}.npy'.format(file_name)), pred_labels)

    def evaluate_sequences(self, ref_seq, pred_seq):
        cmatrix = confusion_matrix(ref_seq,pred_seq, labels=[0,1])
        acc = (cmatrix[0,0]+cmatrix[1,1])/np.sum(np.sum(cmatrix))
        if cmatrix[0,0]==0 or cmatrix[1,1]==0:
            sens = 1
            spec = 1
        else:
            sens = cmatrix[0,0]/(cmatrix[0,0]+cmatrix[0,1])
            spec = cmatrix[1,1]/(cmatrix[1,0]+cmatrix[1,1])
        return acc, sens, spec

    def combine_seqlogitspro(self, seq_logits, seq_files, fold_n=0):
        print('Combining logits for test sequences:')
        seq_cnt = 0
        for file_cnt, file_name in enumerate(seq_files.fname):
            if (seq_cnt+seq_files.nb_seq[file_cnt])>seq_logits.shape[0]:
                break
            if not self._silent:
                print('Combining sequences of file# {}: {}'.format(file_cnt, file_name))
            pred_labels = np.zeros((seq_files.nb_seq[file_cnt],seq_files.nb_frames[file_cnt]))
            pred_multipliers = np.zeros(seq_files.nb_frames[file_cnt])
            pred_multipliers[0:self._seq_hop_len] = 1
            pred_multipliers[(seq_files.nb_seq[file_cnt]*self._seq_hop_len):seq_files.nb_frames[file_cnt]] = 1
            pred_multipliers[self._seq_hop_len:(seq_files.nb_seq[file_cnt]*self._seq_hop_len)] = 0.5
            seq_labels = seq_logits[seq_cnt:(seq_cnt+seq_files.nb_seq[file_cnt]),:,:]
            for file_seq_cnt in range(seq_files.nb_seq[file_cnt]):
                pred_labels[file_seq_cnt,(file_seq_cnt*self._seq_hop_len):(file_seq_cnt*self._seq_hop_len+self._seq_len)] = np.squeeze(seq_labels[file_seq_cnt,:])
            seq_cnt = seq_cnt+seq_files.nb_seq[file_cnt]
            pred_labels = np.sum(pred_labels, axis=0)
            pred_labels = np.multiply(pred_labels,pred_multipliers)
            np.save(os.path.join(self._results_dir, 'test_val_logits_fold{:0>2d}'.format(fold_n+1), '{}.npy'.format(file_name)), pred_labels)