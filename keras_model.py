import os
import sys
import numpy as np
from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Add
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute, Flatten
from keras.layers.recurrent import GRU, LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import keras
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)


keras.backend.set_image_data_format('channels_first')

def get_model(params, network_params):
    x = Input(shape=(2*params['nb_channels'], params['seq_len'], params['nb_freq_bins']))
    spec_cnn = x
    #CNN
    for cnn_cnt, cnn_pool in enumerate(network_params['pool_size']):
        spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'], kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, cnn_pool))(spec_cnn)
        spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)
    #RNN
    spec_rnn = Reshape((params['seq_len'], -1))(spec_cnn)
    for nb_rnn in network_params['rnn_size']:
        spec_rnn = Bidirectional(
            GRU(nb_rnn, activation='tanh', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)
    # FCN - Swalowing Event Detection (SwED)
    swed = spec_rnn
    for nb_fcn in network_params['fcn_size']:
        swed = TimeDistributed(Dense(nb_fcn))(swed)
        swed = Dropout(network_params['dropout_rate'])(swed)
    swed = TimeDistributed(Dense(1))(swed)#TimeDistributed(Dense(params['nb_classes']))(swed)
    swed = Activation('sigmoid', name='sed_out')(swed)

    model = Model(inputs=x, outputs=[swed])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'], metrics=['accuracy'])
    model.summary()
    return model


def get_rawmodel(params, network_params):
    x = Input(shape=(params['nb_channels'], params['seq_len'], params['win_len']))
    sig_cnn = x
    #1DCNN
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][0], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size'][0]))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][1], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size'][1]))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size'][2]))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)
    #RNN
    sig_cnn = Permute((2, 1, 3))(sig_cnn)
    sig_rnn = Reshape((params['seq_len'], -1))(sig_cnn)
    for nb_rnn in network_params['rnn_size']:
        sig_rnn = Bidirectional(
            GRU(nb_rnn, activation='tanh', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
                return_sequences=True),
            merge_mode='mul'
        )(sig_rnn)
    # FCN - Swalowing Event Detection (SwED)
    swed = sig_rnn
    for nb_fcn in network_params['fcn_size']:
        swed = TimeDistributed(Dense(nb_fcn))(swed)
        swed = Dropout(network_params['dropout_rate'])(swed)
    swed = TimeDistributed(Dense(1))(swed)#TimeDistributed(Dense(params['nb_classes']))(swed)
    swed = Activation('sigmoid', name='swed_out')(swed)

    model = Model(inputs=x, outputs=[swed])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'], metrics=['accuracy'])
    model.summary()
    return model

def get_fcmodel(params, network_params):
    x = Input(shape=(2, params['nb_freq_bins'], params['nb_channels']))
    sig_fcn = Permute((2, 3, 1))(x)
    sig_fcn = Flatten()(sig_fcn)
    for nb_fcn in network_params['fcn_size']:
        sig_fcn = Dense(nb_fcn)(sig_fcn)
        sig_fcn = Dropout(network_params['dropout_rate'])(sig_fcn)
    sig_fcn = Dense(1)(sig_fcn)
    swed = Activation('sigmoid', name='swed_out')(sig_fcn)
    model = Model(inputs=x, outputs=[swed])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'], metrics=['accuracy'])
    model.summary()
    return model

def get_fcVGG16model(params, network_params):
    x = Input(shape=(2, params['nb_freq_bins'], params['nb_channels']))
    spec_cnn = x
    #2DCNN
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][0], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][0], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(network_params['pool_size'],1))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][1], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][1], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(network_params['pool_size'],1))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(network_params['pool_size'],1))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(network_params['pool_size'],1))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(network_params['pool_size'],1))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)
    
    sig_fcn = Flatten()(spec_cnn)
    for nb_fcn in network_params['fcn_size']:
        sig_fcn = Dense(nb_fcn)(sig_fcn)
        sig_fcn = Dropout(network_params['dropout_rate'])(sig_fcn)
    sig_fcn = Dense(1)(sig_fcn)
    swed = Activation('sigmoid', name='swed_out')(sig_fcn)
    model = Model(inputs=x, outputs=[swed])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'], metrics=['accuracy'])
    model.summary()
    return model

def get_ssVGG16model(params, network_params):
    x = Input(shape=(2*params['nb_channels'], params['seq_len'], params['nb_freq_bins']))
    spec_cnn = x
    #CNN
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][0], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][0], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size'][0]))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][1], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][1], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size'][1]))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][2], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][2], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][2], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size'][2]))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size'][3]))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size'][4]))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)

    #RNN
    
    spec_rnn = Reshape((params['seq_len'], -1))(spec_cnn)
    for nb_rnn in network_params['rnn_size']:
        spec_rnn = Bidirectional(
            GRU(nb_rnn, activation='tanh', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)
    # FCN - Swalowing Event Detection (SwED)
    swed = spec_rnn
    for nb_fcn in network_params['fcn_size']:
        swed = TimeDistributed(Dense(nb_fcn))(swed)
        swed = Dropout(network_params['dropout_rate'])(swed)
    swed = TimeDistributed(Dense(1))(swed)#TimeDistributed(Dense(params['nb_classes']))(swed)
    swed = Activation('sigmoid', name='sed_out')(swed)

    model = Model(inputs=x, outputs=[swed])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'], metrics=['accuracy'])
    model.summary()
    return model


def get_rwVGG16model(params, network_params):
    x = Input(shape=(params['nb_channels'], params['seq_len'], params['win_len']))
    sig_cnn = x
    #1DCNN
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][0], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][0], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size']))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)

    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][1], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][1], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size']))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)

    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size']))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)

    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size']))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)

    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size']))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)
    sig_cnn = Permute((2, 1, 3))(sig_cnn)
    
    #RNN
    
    sig_rnn = Reshape((params['seq_len'], -1))(sig_cnn)
    for nb_rnn in network_params['rnn_size']:
        sig_rnn = Bidirectional(
            GRU(nb_rnn, activation='tanh', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
                return_sequences=True),
            merge_mode='mul'
        )(sig_rnn)
    # FCN - Swalowing Event Detection (SwED)
    swed = sig_rnn
    for nb_fcn in network_params['fcn_size']:
        swed = TimeDistributed(Dense(nb_fcn))(swed)
        swed = Dropout(network_params['dropout_rate'])(swed)
    swed = TimeDistributed(Dense(1))(swed)#TimeDistributed(Dense(params['nb_classes']))(swed)
    swed = Activation('sigmoid', name='swed_out')(swed)

    model = Model(inputs=x, outputs=[swed])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'], metrics=['accuracy'])
    model.summary()
    return model

def get_ssResVGG16model(params, network_params):
    x = Input(shape=(2*params['nb_channels'], params['seq_len'], params['nb_freq_bins']))
    spec_cnn = x
    
    #CNN_unit_1
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][0], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    
    ##CNN_unit_2
    spec_cnn_r = spec_cnn
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][0], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][0], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Add()([spec_cnn, spec_cnn_r])
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size'][0]))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    #CNN_unit_3
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][1], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)

    #CNN_unit_4
    spec_cnn_r = spec_cnn
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][1], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][1], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Add()([spec_cnn, spec_cnn_r])
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size'][1]))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    #CNN_unit_5
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][2], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)

    #CNN_unit_6
    spec_cnn_r = spec_cnn
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][2], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][2], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][2], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Add()([spec_cnn, spec_cnn_r])
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size'][2]))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    #CNN_unit_7
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    
    #CNN_unit_8
    spec_cnn_r = spec_cnn
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Add()([spec_cnn, spec_cnn_r])
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size'][3]))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    #CNN_unit_9
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)

    #CNN_unit_10
    spec_cnn_r = spec_cnn
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'][3], kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Add()([spec_cnn, spec_cnn_r])
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size'][4]))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)
    
    spec_cnn = Permute((2, 1, 3))(spec_cnn)

    #RNN
    
    spec_rnn = Reshape((params['seq_len'], -1))(spec_cnn)
    for nb_rnn in network_params['rnn_size']:
        spec_rnn = Bidirectional(
            GRU(nb_rnn, activation='tanh', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)
    # FCN - Swalowing Event Detection (SwED)
    swed = spec_rnn
    for nb_fcn in network_params['fcn_size']:
        swed = TimeDistributed(Dense(nb_fcn))(swed)
        swed = Dropout(network_params['dropout_rate'])(swed)
    swed = TimeDistributed(Dense(1))(swed)#TimeDistributed(Dense(params['nb_classes']))(swed)
    swed = Activation('sigmoid', name='sed_out')(swed)

    model = Model(inputs=x, outputs=[swed])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'], metrics=['accuracy'])
    model.summary()
    return model

def get_rwResVGG16model(params, network_params):
    x = Input(shape=(params['nb_channels'], params['seq_len'], params['win_len']))
    sig_cnn = x
    
    #CNN_unit_1
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][0], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)

    #CNN_unit_2
    sig_cnn_r = sig_cnn
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][0], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][0], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Add()([sig_cnn, sig_cnn_r])
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size']))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)

    #CNN_unit_3
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][1], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    
    #CNN_unit_4
    sig_cnn_r = sig_cnn
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][1], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][1], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Add()([sig_cnn, sig_cnn_r])
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size']))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)

    #CNN_unit_5
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)

    #CNN_unit_6
    sig_cnn_r = sig_cnn
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Add()([sig_cnn, sig_cnn_r])
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size']))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)

    #CNN_unit_7
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)

    #CNN_unit_8
    sig_cnn_r = sig_cnn
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Add()([sig_cnn, sig_cnn_r])
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size']))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)

    #CNN_unit_9
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)

    #CNN_unit_10
    sig_cnn_r = sig_cnn
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(1,5), padding='same')(sig_cnn)
    sig_cnn = BatchNormalization()(sig_cnn)
    sig_cnn = Activation('relu')(sig_cnn)
    sig_cnn = Add()([sig_cnn, sig_cnn_r])
    sig_cnn = MaxPooling2D(pool_size=(1,network_params['pool_size']))(sig_cnn)
    sig_cnn = Dropout(network_params['dropout_rate'])(sig_cnn)
    sig_cnn = Permute((2, 1, 3))(sig_cnn)
    
    #RNN
    
    sig_rnn = Reshape((params['seq_len'], -1))(sig_cnn)
    for nb_rnn in network_params['rnn_size']:
        sig_rnn = Bidirectional(
            GRU(nb_rnn, activation='tanh', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
                return_sequences=True),
            merge_mode='mul'
        )(sig_rnn)
    # FCN - Swalowing Event Detection (SwED)
    swed = sig_rnn
    for nb_fcn in network_params['fcn_size']:
        swed = TimeDistributed(Dense(nb_fcn))(swed)
        swed = Dropout(network_params['dropout_rate'])(swed)
    swed = TimeDistributed(Dense(1))(swed)#TimeDistributed(Dense(params['nb_classes']))(swed)
    swed = Activation('sigmoid', name='swed_out')(swed)

    model = Model(inputs=x, outputs=[swed])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'], metrics=['accuracy'])
    model.summary()
    return model

def get_fcResVGG16model(params, network_params):
    x = Input(shape=(2, params['nb_freq_bins'], params['nb_channels']))
    spec_cnn = x
    
    #CNN_unit_1
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][0], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)

    #CNN_unit_2
    spec_cnn_r = spec_cnn
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][0], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][0], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Add()([spec_cnn, spec_cnn_r])
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(network_params['pool_size'],1))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    #CNN_unit_3
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][1], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)

    #CNN_unit_4
    spec_cnn_r = spec_cnn
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][1], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][1], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Add()([spec_cnn, spec_cnn_r])
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(network_params['pool_size'],1))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    #CNN_unit_5
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)

    #CNN_unit_6
    spec_cnn_r = spec_cnn
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][2], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Add()([spec_cnn, spec_cnn_r])
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(network_params['pool_size'],1))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    #CNN_unit_7
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)

    #CNN_unit_8
    spec_cnn_r = spec_cnn
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Add()([spec_cnn, spec_cnn_r])
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(network_params['pool_size'],1))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    #CNN_unit_9
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)

    #CNN_unit_10
    spec_cnn_r = spec_cnn
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Conv2D(filters=network_params['nb_cnn1d_filt'][3], kernel_size=(5,3), padding='same')(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = Add()([spec_cnn, spec_cnn_r])
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D(pool_size=(network_params['pool_size'],1))(spec_cnn)
    spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)
    
    sig_fcn = Flatten()(spec_cnn)
    for nb_fcn in network_params['fcn_size']:
        sig_fcn = Dense(nb_fcn)(sig_fcn)
        sig_fcn = Dropout(network_params['dropout_rate'])(sig_fcn)
    sig_fcn = Dense(1)(sig_fcn)
    swed = Activation('sigmoid', name='swed_out')(sig_fcn)
    model = Model(inputs=x, outputs=[swed])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'], metrics=['accuracy'])
    model.summary()
    return model
