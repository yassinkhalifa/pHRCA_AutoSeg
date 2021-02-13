import os
import sys
import numpy as np
import matplotlib.pyplot as plot
from keras.models import load_model
from keras.callbacks import CSVLogger, EarlyStopping
from keras_model import get_fcmodel
import h5py
import time
from pathlib import Path
import pandas as pd
import warnings
from helper_classes import data_splitter, DataGenerator_noseq
from sklearn.metrics import confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=DeprecationWarning)

params = {
        'log_dir':'./logs',
        'data_dir':'./ftrs_prcsd/',
        'results_dir':'./results',
        'win_len':800,
        'hop_len':400,
        'nb_channels':3,
        'nb_classes':2,
        'nb_folds':10,
        'nb_epochs':100,
        'nb_freq_bins':512,
        'batch_size':32,
        'patience':10
    }
network_params = {
        'fcn_size':[512, 512, 512, 512],
        'dropout_rate':0.0
    }

def main(params, network_params):
    print('\n\n----------------------------------------------------------------------------------------------------')
    print('----------------------------------- Splitting data into {:0>2d} folds -----------------------------------'.format(params['nb_folds']))
    print('----------------------------------------------------------------------------------------------------')
    data_splitter_cls = data_splitter(data_dir=params['data_dir'], win_len=params['win_len'], hop_len=params['hop_len'], nb_channels=params['nb_channels'], nb_classes=params['nb_classes'], validate=False, nb_folds=params['nb_folds'], silent=True)
    data_splitter_cls.perform()
    Path(os.path.join(params['log_dir'])).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(params['results_dir'])).mkdir(parents=True, exist_ok=True)
    acc = np.zeros((params['nb_folds'],1))
    sens = np.zeros((params['nb_folds'],1))
    spec = np.zeros((params['nb_folds'],1))
    for fold_cnt in range(params['nb_folds']):
        print('\n\n----------------------------------------------------------------------------------------------------')
        print('------------------------------------------    fold: {:0>2d}    ------------------------------------------'.format(fold_cnt+1))
        print('----------------------------------------------------------------------------------------------------')
        model_name = os.path.join(params['log_dir'],'fold_{}_best_model.h5'.format(fold_cnt+1))
        csv_logger = CSVLogger(filename=os.path.join(params['log_dir'],'fold_{}_log.csv'.format(fold_cnt+1)))
        early_stopper = EarlyStopping(monitor='loss', min_delta=0, mode='min')
        data_splitter_cls.noseq_prepare(fold_n=fold_cnt, run_type='train')
        data_splitter_cls.noseq_prepare(fold_n=fold_cnt, run_type='val')
        data_splitter_cls.noseq_prepare(fold_n=fold_cnt, run_type='test')
        print('Loading training dataset:')
        train_data_gen = DataGenerator_noseq(data_dir=params['data_dir'], shuffle=False, win_len=params['win_len'], hop_len=params['hop_len'], nb_channels=params['nb_channels'], nb_classes=params['nb_classes'], n_fold=fold_cnt, batch_size=params['batch_size'], run_type='train')
        val_data_gen = DataGenerator_noseq(data_dir=params['data_dir'], shuffle=False, win_len=params['win_len'], hop_len=params['hop_len'], nb_channels=params['nb_channels'], nb_classes=params['nb_classes'], n_fold=fold_cnt, batch_size=params['batch_size'], run_type='val')
        test_data_gen = DataGenerator_noseq(data_dir=params['data_dir'], shuffle=False, win_len=params['win_len'], hop_len=params['hop_len'], nb_channels=params['nb_channels'], nb_classes=params['nb_classes'], n_fold=fold_cnt, batch_size=params['batch_size'], run_type='test')
        model = get_fcmodel(params, network_params)
        hist = model.fit(x=train_data_gen, validation_data=val_data_gen, epochs=params['nb_epochs'], workers=10, use_multiprocessing=True, callbacks=[csv_logger]) #early_stopper,csv_logger])
        model.save(model_name)

        print('\nLoading the best model and predicting results on the testing data')
        model = load_model(model_name)
        pred_test = model.predict(x=test_data_gen, workers=2, use_multiprocessing=False)
        pred_test_logits = pred_test
        pred_test = pred_test>0.5
        test_h5 = h5py.File(os.path.join(test_data_gen._data_dir, 'folds_h5', '{}_metadata_{}.hdf5'.format('test', fold_cnt+1)))
        ref_test = np.asarray(test_h5['frame_labels'])
        test_h5.close()
        Path(os.path.join(params['results_dir'], 'test_val')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(params['results_dir'], 'test_val_logits')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(params['results_dir'], 'ref_val')).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(params['results_dir'], 'test_val', 'pred_fold_{:0>2d}.npy'.format(fold_cnt+1)), pred_test)
        np.save(os.path.join(params['results_dir'], 'test_val_logits', 'logits_fold_{:0>2d}.npy'.format(fold_cnt+1)), pred_test_logits)
        np.save(os.path.join(params['results_dir'], 'ref_val', 'ref_fold_{:0>2d}.npy'.format(fold_cnt+1)), ref_test)
        cmatrix = confusion_matrix(ref_test[:pred_test.shape[0]], pred_test, labels=[0,1])
        acc[fold_cnt] = (cmatrix[0,0]+cmatrix[1,1])/np.sum(np.sum(cmatrix))
        if cmatrix[0,0]==0 or cmatrix[1,1]==0:
            sens[fold_cnt] = 1
            spec[fold_cnt] = 1
        else:
            sens[fold_cnt] = cmatrix[0,0]/(cmatrix[0,0]+cmatrix[0,1])
            spec[fold_cnt] = cmatrix[1,1]/(cmatrix[1,0]+cmatrix[1,1])
    np.save(os.path.join(params['results_dir'], 'qmeasures_acc.npy'), acc)
    np.save(os.path.join(params['results_dir'], 'qmeasures_sens.npy'), sens)
    np.save(os.path.join(params['results_dir'], 'qmeasures_spec.npy'), spec)


if __name__ == "__main__":
    main(params, network_params)
