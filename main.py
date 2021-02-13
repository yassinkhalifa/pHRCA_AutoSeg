import os
import sys
import numpy as np
import matplotlib.pyplot as plot
from keras.models import load_model
from keras.callbacks import CSVLogger, EarlyStopping
from keras_model import get_model
import h5py
import time
from pathlib import Path
import pandas as pd
import warnings
from helper_classes import data_splitter, DataGenerator, evaluation_metrics
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=DeprecationWarning)

params = {
        'log_dir':'./logs',
        'data_dir':'./ftrs_prcsd/',
        'results_dir':'./results',
        'win_len':800,
        'hop_len':400,
        'seq_len':10,
        'seq_hop_len':5,
        'nb_channels':3,
        'nb_classes':2,
        'nb_folds':10,
        'nb_epochs':100,
        'nb_freq_bins':512,
        'batch_size':16,
        'patience':10
    }
network_params = {
        'nb_cnn2d_filt':64,
        'pool_size':[8, 8, 4],
        'rnn_size':[128, 128],
        'fcn_size':[128, 128],
        'dropout_rate':0.0
    }


def main(params, network_params):
    print('\n\n----------------------------------------------------------------------------------------------------')
    print('----------------------------------- Splitting data into {:0>2d} folds -----------------------------------'.format(params['nb_folds']))
    print('----------------------------------------------------------------------------------------------------')
    evaluation_cls = evaluation_metrics(results_dir=params['results_dir'], win_len=params['win_len'], hop_len=params['hop_len'], seq_len=params['seq_len'], seq_hop_len=params['seq_hop_len'], nb_channels=params['nb_channels'], nb_classes=params['nb_classes'], silent=True)
    data_splitter_cls = data_splitter(data_dir=params['data_dir'], win_len=params['win_len'], hop_len=params['hop_len'], seq_len=params['seq_len'], seq_hop_len=params['seq_hop_len'], nb_channels=params['nb_channels'], nb_classes=params['nb_classes'], validate=False, nb_folds=params['nb_folds'], silent=True)
    data_splitter_cls.perform()
    Path(os.path.join(params['log_dir'])).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(params['results_dir'])).mkdir(parents=True, exist_ok=True)
    for fold_cnt in range(params['nb_folds']):
        print('\n\n----------------------------------------------------------------------------------------------------')
        print('------------------------------------------    fold: {:0>2d}    ------------------------------------------'.format(fold_cnt+1))
        print('----------------------------------------------------------------------------------------------------')
        model_name = os.path.join(params['log_dir'],'fold_{}_best_model.h5'.format(fold_cnt+1))
        csv_logger = CSVLogger(filename=os.path.join(params['log_dir'],'fold_{}_log.csv'.format(fold_cnt+1)))
        early_stopper = EarlyStopping(monitor='loss', min_delta=0, mode='min')
        data_splitter_cls.prepare(fold_n=fold_cnt, run_type='train')
        data_splitter_cls.prepare(fold_n=fold_cnt, run_type='val')
        data_splitter_cls.prepare(fold_n=fold_cnt, run_type='test')
        print('Loading training dataset:')
        train_data_gen = DataGenerator(data_dir=params['data_dir'], shuffle=False, win_len=params['win_len'], hop_len=params['hop_len'], seq_len=params['seq_len'], seq_hop_len=params['seq_hop_len'], nb_channels=params['nb_channels'], nb_classes=params['nb_classes'], n_fold=fold_cnt, batch_size=params['batch_size'], run_type='train')
        val_data_gen = DataGenerator(data_dir=params['data_dir'], shuffle=False, win_len=params['win_len'], hop_len=params['hop_len'], seq_len=params['seq_len'], seq_hop_len=params['seq_hop_len'], nb_channels=params['nb_channels'], nb_classes=params['nb_classes'], n_fold=fold_cnt, batch_size=params['batch_size'], run_type='val')
        test_data_gen = DataGenerator(data_dir=params['data_dir'], shuffle=False, win_len=params['win_len'], hop_len=params['hop_len'], seq_len=params['seq_len'], seq_hop_len=params['seq_hop_len'], nb_channels=params['nb_channels'], nb_classes=params['nb_classes'], n_fold=fold_cnt, batch_size=params['batch_size'], run_type='test')
        model = get_model(params, network_params)
        #for epoch_cnt in range(params['nb_epochs']):
            #print('Epoch No. {}'.format(epoch_cnt+1))
        hist = model.fit(x=train_data_gen, validation_data=val_data_gen, epochs=params['nb_epochs'], workers=10, use_multiprocessing=True, callbacks=[csv_logger]) #early_stopper,csv_logger])
        model.save(model_name)

        print('\nLoading the best model and predicting results on the testing data')
        model = load_model(model_name)
        pred_test = model.predict(x=test_data_gen, workers=2, use_multiprocessing=False)
        pred_test_logits = pred_test
        pred_test = pred_test>0.5
        test_data_records = pd.read_csv(os.path.join(test_data_gen._data_dir, 'folds_metadata', '{}_metadata_{}.csv'.format(test_data_gen._run_type, fold_cnt+1)))
        test_h5 = h5py.File(os.path.join(test_data_gen._data_dir, 'folds_h5', '{}_metadata_{}.hdf5'.format('test', fold_cnt+1)))
        ref_test = np.asarray(test_h5['seq_labels'])
        test_h5.close()
        Path(os.path.join(params['results_dir'], 'test_val_fold{:0>2d}'.format(fold_cnt+1))).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(params['results_dir'], 'test_val_logits_fold{:0>2d}'.format(fold_cnt+1))).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(params['results_dir'], 'ref_val_fold{:0>2d}'.format(fold_cnt+1))).mkdir(parents=True, exist_ok=True)
        evaluation_cls.combine_seqlogits(pred_test,test_data_records,run_type='test',fold_n=fold_cnt)
        evaluation_cls.combine_seqlogits(ref_test,test_data_records,run_type='ref',fold_n=fold_cnt)
        evaluation_cls.combine_seqlogitspro(pred_test_logits,test_data_records,fold_n=fold_cnt)
        nb_test_files = os.listdir(os.path.join(params['results_dir'], 'test_val_fold{:0>2d}'.format(fold_cnt+1)))
        acc = np.zeros((len(nb_test_files),1))
        sens = np.zeros((len(nb_test_files),1))
        spec = np.zeros((len(nb_test_files),1))
        fold_clogits = []
        fold_cgtruth = []
        print('Model evaluation for fold#{}:'.format(fold_cnt+1))
        for testfile_cnt, testfile in enumerate(nb_test_files):
            print('Evaluating segmentation of file# {}: {}'.format(testfile_cnt+1, testfile))
            acc[testfile_cnt], sens[testfile_cnt], spec[testfile_cnt] = evaluation_cls.evaluate_sequences(np.load(os.path.join(params['results_dir'], 'ref_val_fold{:0>2d}'.format(fold_cnt+1), testfile)),np.load(os.path.join(params['results_dir'], 'test_val_fold{:0>2d}'.format(fold_cnt+1), testfile)))
            fold_clogits = np.concatenate((fold_clogits, np.load(os.path.join(params['results_dir'], 'test_val_logits_fold{:0>2d}'.format(fold_cnt+1), testfile))))
            fold_cgtruth = np.concatenate((fold_cgtruth, np.load(os.path.join(params['results_dir'], 'ref_val_fold{:0>2d}'.format(fold_cnt+1), testfile))))
        np.save(os.path.join(params['results_dir'], 'fold_{:0>2d}_aggregate_logits.npy'.format(fold_cnt+1)), fold_clogits)
        np.save(os.path.join(params['results_dir'], 'fold_{:0>2d}_aggregate_gtruth.npy'.format(fold_cnt+1)), fold_cgtruth)
        np.save(os.path.join(params['results_dir'], 'fold_{:0>2d}_qmeasures_acc.npy'.format(fold_cnt+1)), acc)
        np.save(os.path.join(params['results_dir'], 'fold_{:0>2d}_qmeasures_sens.npy'.format(fold_cnt+1)), sens)
        np.save(os.path.join(params['results_dir'], 'fold_{:0>2d}_qmeasures_spec.npy'.format(fold_cnt+1)), spec)


if __name__ == "__main__":
    main(params, network_params)
