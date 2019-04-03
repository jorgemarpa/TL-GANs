import numpy as np
import json
import types

import tensorflow as tf

import keras.layers
from keras.layers import (Input, Dense, TimeDistributed, merge, Lambda,
                          concatenate, Bidirectional, LSTM, GRU, RNN)
from keras.layers.core import *
from keras.layers.convolutional import *
#from keras.layers.recurrent import *

from keras import backend as K
from keras.activations import relu
from keras.models import Sequential, Model

relu_size = 16
dict_rnn = {'RNN': RNN, 'GRU': GRU, 'LSTM': LSTM}


## ########################################################################## ##
# FUNCTIONS
## ########################################################################## ##

## -------------------------------------------------------------------------- ##
# RESIDUAL BLOCK
## -------------------------------------------------------------------------- ##
def residual_block(x, sizenet, m_dilation_rate, m_stack, causal=False,
                   m_activation='wavenet', drop_frac=0.25, kernel_size=3, name=''):
    """ ---------------------------------------------------------
        @summary: residual block.
        @note: "WaveNet: A Generative Model for Raw Audio",
                                van den Oord et al, 2016, [arXiv:1609.03499v2]
        @param x: Keras layer.
        @param sizenet: int, size of the network.
        @return m_dilation_rate: int, dilation rate
        @param m_stack: int, number of stacks/layers
        @param causal: boolean, if causal padding.
        @param m_activation: str, activation function.
        @param drop_frac: float(<1), fraction of dropout in keras archi
        @param kernel_size: int, size of 1D conv filters.
        @param name: str, layer name.
        @return (x_residual, x): tuple, residual model layer and the skip connection.
    ---------------------------------------------------------- """

    x_init = x

    m_padding = 'causal' if causal else 'same'

    x = Conv1D(filters=sizenet,
               kernel_size=kernel_size,
               dilation_rate=m_dilation_rate,
               name=name +
               'B{}_L{}_dilC{}_N{}'.format(
                   m_stack, m_dilation_rate, kernel_size, sizenet),
               padding=m_padding)(x)

    if m_activation == 'wavenet':
        x = keras.layers.multiply([Activation('tanh', name=name + 'B{}_L{}_activation_tanh'.format(m_stack, m_dilation_rate))(x),
                                   Activation('sigmoid', name=name + 'B{}_L{}_activation_sigmoid'.format(m_stack, m_dilation_rate))(x)],
                                  name=name + 'B{}_L{}_activation_multiply'.format(m_stack, m_dilation_rate))
    else:
        x = Activation(m_activation, name=name +
                       'B{}_L{}_activation_{}'.format(m_stack, m_dilation_rate, m_activation))(x)

    x = SpatialDropout1D(rate=drop_frac,
                         name=name + 'B{}_L{}_sdrop{}'.format(m_stack, m_dilation_rate, int(drop_frac * 100)))(x)

    x = Conv1D(filters=sizenet,
               kernel_size=1,
               padding='same',
               name=name + 'B{}_L{}_dilC{}_N{}'.format(m_stack, m_dilation_rate, 1, sizenet))(x)

    x_residual = keras.layers.add([x_init, x],
                                  name=name + 'B{}_L{}_add'.format(m_stack, m_dilation_rate))

    return x_residual, x


## ########################################################################## ##
# AUTO-ENCODERS
## ########################################################################## ##

## -------------------------------------------------------------------------- ##
## 1 - RNN
## -------------------------------------------------------------------------- ##
def net_RNN_pb(input_dimension, output_size, sizenet,
               nb_passbands=1, max_lenght=None,
               num_layers=1,
               drop_frac=0.0,
               model_type='LSTM',
               bidirectional=True,
               return_sequences=False,
               add_meta=None,
               add_dense=False,
               last_activation='tanh',
               **kwargs):

    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}

    layer = dict_rnn[model_type]

    meta_input = Input(shape=(add_meta,),
                       name='meta_input') if add_meta is not None else None

    main_input_list = []
    for j in range(nb_passbands):
        if max_lenght is None:
            ndatapoints = None
        else:
            ndatapoints = max_lenght if isinstance(
                max_lenght, int) else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension),
                           name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)

    # --------------- NET --------------- #
    x_list = []
    for j in range(nb_passbands):
        x = main_input_list[j]
        for i in range(num_layers):
            wrapper = Bidirectional if bidirectional else lambda x: x
            x = wrapper(layer(units=sizenet,
                              name='encode_{}_pb{}'.format(i, j),
                              return_sequences=True if return_sequences else (i < num_layers - 1)))(x)
            if drop_frac > 0.0:
                x = Dropout(rate=drop_frac,
                            name='drop_encode_pb{}_x{}'.format(j, i))(x)

        ## Output per band ##
        # if return_sequences:
        #    x = TimeDistributed(Dense(units = output_size, activation = 'softmax'), name = 'pb{}_time_dist'.format(j))(x)
        # else:
        #    x = Dense(units = output_size, activation ='softmax',  name='pb{}_softmax_dense'.format(j))(x)

        x_list.append(x)

    ## Merge ##
    x_merged = concatenate(x_list, name='concat',
                           axis=1) if nb_passbands > 1 else x_list[0]
    if not return_sequences:
        if (nb_passbands > 1):
            x_merged = concatenate([x_merged, meta_input]) if (
                add_meta is not None) else x_merged
            x_merged = Dense(units=relu_size, activation='relu',
                             name='relu_dense')(x_merged)
            if add_dense:
                x_merged = Dense(
                    units=relu_size // 2, activation='relu', name='relu_dense_2')(x_merged)

            x_merged = Dense(units=output_size, activation='softmax',
                             name='softmax_dense')(x_merged)

        if (nb_passbands == 1):
            x_merged = concatenate([x_merged, meta_input]) if (
                add_meta is not None) else x_merged
            x_merged = Dense(units=relu_size, activation='relu',
                             name='relu_dense')(x_merged)
            if add_dense:
                x_merged = Dense(
                    units=relu_size // 2, activation='relu', name='relu_dense_2')(x_merged)

            x_merged = Dense(units=output_size, activation='softmax',
                             name='softmax_dense')(x_merged)

    else:
        if (nb_passbands > 1):
            x_merged = Dense(units=relu_size, activation='relu',
                             name='relu_dense')(x_merged)
            # if add_dense:
            #    x_merged = Dense(units = relu_size//2, activation ='relu',  name='relu_dense_2')(x_merged)

        x_merged = TimeDistributed(Dense(units=output_size, activation=last_activation),
                                   name='time_dist_{}_dense'.format(last_activation))(x_merged)

    '''
    if return_sequences:
        x_merged = TimeDistributed(Dense(units = output_size, activation = 'softmax'), name = 'time_dist')(x_merged)
    else:
        x_merged = concatenate([x_merged, meta_input]) if add_meta is not None else x_merged
        x_merged = Dense(units = output_size, activation ='softmax',  name='softmax_dense')(x_merged)
     '''

    # --------------- MODEL --------------- #
    input_layer = main_input_list + \
        [meta_input] if add_meta is not None else main_input_list
    output_layer = x_merged
    m_model = Model(input_layer, output_layer)

    param_str = '{}_pb{}_n{}_x{}_drop{}_out{}'.format(model_type, nb_passbands,
                                                      sizenet, num_layers,
                                                      int(drop_frac * 100),
                                                      output_size)
    if bidirectional:
        param_str += '_bidir'

    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict


## -------------------------------------------------------------------------- ##
# 2 - TEMPORAL CONVO NETWORK
## -------------------------------------------------------------------------- ##
def net_tCNN_pb(input_dimension, output_size, sizenet, kernel_size,
                nb_passbands=1, max_lenght=None,
                causal=False,
                num_layers=1,
                drop_frac=0.25,
                m_reductionfactor=2,
                m_activation='wavenet',
                return_sequences=True,
                add_meta=None,
                add_dense=False,
                **kwargs):

    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}

    meta_input = Input(shape=(add_meta,),
                       name='meta_input') if add_meta is not None else None

    main_input_list = []
    for j in range(nb_passbands):
        if max_lenght is None:
            ndatapoints = None
        else:
            ndatapoints = max_lenght if isinstance(
                max_lenght, int) else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension),
                           name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)

    m_padding = 'causal' if causal else 'same'

    # --------------- NET --------------- #
    x_list = []
    for j in range(nb_passbands):
        x = main_input_list[j]

        for i in range(num_layers):

            ## Convo ##
            x = Conv1D(filters=sizenet,  # //(2**(i)),
                       kernel_size=kernel_size,
                       #activation = 'relu',
                       input_shape=(max_lenght, input_dimension),
                       padding=m_padding,
                       name='pb{}_x{}_conv1d'.format(j, i))(x)

            ## Activation ##
            if m_activation == 'wavenet':
                x = keras.layers.multiply([Activation('tanh', name='pb{}_x{}_activation_tanh'.format(j, i))(x),
                                           Activation('sigmoid', name='pb{}_x{}_activation_sigmoid'.format(j, i))(x)],
                                          name='pb{}_x{}_activation_multiply'.format(j, i))
            else:
                x = Activation(m_activation, name='pb{}_x{}_activation_{}'.format(
                    j, i, m_activation))(x)

            ## Pooling ##
            if m_reductionfactor > 0:
                x = MaxPooling1D(pool_size=m_reductionfactor, name='pb{}_x{}_maxpool{}'.format(
                    j, i, m_reductionfactor))(x)

            ## Dropout ##
            if drop_frac > 0:
                x = SpatialDropout1D(drop_frac, name='pb{}_x{}_sdrop{}'.format(
                    j, i, int(drop_frac * 100)))(x)

        ## Output per band ##
       # if return_sequences:
       #     x = TimeDistributed(Dense(units = output_size, activation = 'softmax', name='pb{}_softmax'.format(j)), name = 'pb{}_time_dist'.format(j))(x)
       # else:
       #     x = Dense(units = output_size, activation ='softmax', name= 'pb{}_softmax'.format(j))(x)
       #     x = Lambda(lambda y: y[:, -1, :], name='pb{}_embedding'.format(j))(x)
        if not return_sequences:
            x = Lambda(lambda y: y[:, -1, :],
                       name='pb{}_embedding'.format(j))(x)

        x_list.append(x)

    ## Merge ##
    x_merged = concatenate(x_list, name='concat',
                           axis=1) if nb_passbands > 1 else x_list[0]
    if not return_sequences:
        if (nb_passbands > 1):
            x_merged = concatenate([x_merged, meta_input]) if (
                add_meta is not None) else x_merged
            x_merged = Dense(units=relu_size, activation='relu',
                             name='relu_dense')(x_merged)
            if add_dense:
                x_merged = Dense(
                    units=relu_size // 2, activation='relu', name='relu_dense_2')(x_merged)

            x_merged = Dense(units=output_size, activation='softmax',
                             name='softmax_dense')(x_merged)

        if (nb_passbands == 1):
            x_merged = concatenate([x_merged, meta_input]) if (
                add_meta is not None) else x_merged
            x_merged = Dense(units=relu_size, activation='relu',
                             name='relu_dense')(x_merged)
            if add_dense:
                x_merged = Dense(
                    units=relu_size // 2, activation='relu', name='relu_dense_2')(x_merged)

            x_merged = Dense(units=output_size, activation='softmax',
                             name='softmax_dense')(x_merged)

    else:
        if (nb_passbands > 1):
            x_merged = Dense(units=relu_size, activation='relu',
                             name='relu_dense')(x_merged)
            # if add_dense:
            #   x_merged = Dense(units = relu_size//2, activation ='relu',  name='relu_dense_2')(x_merged)

        x_merged = TimeDistributed(Dense(
            units=output_size, activation='linear'), name='time_dist'.format(j))(x_merged)

    # --------------- MODEL --------------- #
    input_layer = main_input_list + \
        [meta_input] if add_meta is not None else main_input_list
    output_layer = x_merged
    model = Model(input_layer, output_layer)

    param_str = 'tCNN_pb{}_n{}_x{}_drop{}_cv{}_out{}'.format(nb_passbands,
                                                             sizenet, num_layers,
                                                             int(drop_frac * 100),
                                                             kernel_size, output_size)
    if m_activation == 'wavenet':
        param_str += '_aW'
    if causal:
        param_str += '_causal'

    print('>>>> param_str = ', param_str)
    return model, param_str, param_dict


## -------------------------------------------------------------------------- ##
## 3 - DILATED-TCN
## -------------------------------------------------------------------------- ##
def net_dTCN_pb(input_dimension, output_size, sizenet, n_stacks,
                list_dilations=None, max_dilation=None,
                output_size_cw=None,
                nb_passbands=1, max_lenght=None,
                causal=False,
                drop_frac=0.25,
                config_wavenet=True,
                m_activation='wavenet',
                use_skip_connections=True,
                kernel_size=3, kernel_wavenet=1,
                return_sequences=False,
                add_meta=None,
                add_dense=False,
                **kwargs):

    if output_size_cw is None:
        output_size_cw = output_size

    if list_dilations is None:
        list_dilations = list(range(max_dilation))

    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}

    dilation_depth = len(list_dilations)
    m_dilations = [2**i for i in list_dilations]

    meta_input = Input(shape=(add_meta,),
                       name='meta_input') if add_meta is not None else None

    main_input_list = []  # aux_input_list=[]
    for j in range(nb_passbands):
        if max_lenght is None:
            ndatapoints = None
        else:
            ndatapoints = max_lenght if isinstance(max_lenght,
                                                   int) else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension),
                           name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)

    m_padding = 'causal' if causal else 'same'

    # --------------- NET --------------- #
    x_list = []
    for j in range(nb_passbands):
        x = main_input_list[j]

        ## Convo ##
        x = Conv1D(filters=sizenet,
                   kernel_size=kernel_size,
                   padding=m_padding,
                   name='pb{}_conv1d'.format(j))(x)

        ## Residuals ##
        skip_connections = []
        for m_stack in range(n_stacks):
            for m_dilation_rate in m_dilations:
                x, skip_x = residual_block(x, sizenet, m_dilation_rate,
                                           m_stack, causal,
                                           m_activation, drop_frac,
                                           kernel_size, name='pb{}_'.format(j))
                skip_connections.append(skip_x)

        if use_skip_connections:
            x = keras.layers.add(skip_connections,
                                 name='pb{}_add_skip_connections'.format(j))

        ## Wavenet config ##
        x = Activation('relu', name='pb{}_relu'.format(j))(x)
        if config_wavenet:
            x = Conv1D(filters=sizenet,
                       kernel_size=kernel_wavenet,
                       padding='same',
                       name='pb{}_n{}_cv{}_cw'.format(j, sizenet,
                                                      kernel_wavenet))(x)
            #
            x = Activation('relu', name='pb{}_relu_cw'.format(j))(x)
            #
            x = Conv1D(filters=output_size_cw,
                       kernel_size=kernel_wavenet,
                       padding='same',
                       name='pb{}_n{}_cv{}_cw'.format(j, output_size_cw,
                                                      kernel_wavenet))(x)
        else:
            x = Dense(units=output_size_cw, name='pb{}_dense'.format(j))(x)

        ## Output per band ##
        # if return_sequences:
        #    x = TimeDistributed(Activation('softmax', name = 'pb{}_softmax'.format(j)), name = 'pb{}_time_dist'.format(j))(x)
        # else:
        #    x = Activation('softmax', name = 'pb{}_softmax'.format(j))(x)
        #    x = Lambda(lambda y: y[:, -1, :], name='pb{}_embedding'.format(j))(x)
        if not return_sequences:
            x = Lambda(lambda y: y[:, -1, :],
                       name='pb{}_embedding'.format(j))(x)

        x_list.append(x)

    ## Merge ##
    x_merged = concatenate(x_list, name='concat',
                           axis=1) if nb_passbands > 1 else x_list[0]
    if not return_sequences:
        if (nb_passbands > 1):
            x_merged = concatenate([x_merged, meta_input]) if (
                add_meta is not None) else x_merged
            x_merged = Dense(units=relu_size, activation='relu',
                             name='relu_dense')(x_merged)
            if add_dense:
                x_merged = Dense(
                    units=relu_size // 2, activation='relu', name='relu_dense_2')(x_merged)

            x_merged = Dense(units=output_size, activation='softmax',
                             name='softmax_dense')(x_merged)

        if (nb_passbands == 1):
            x_merged = concatenate([x_merged, meta_input]) if (
                add_meta is not None) else x_merged
            x_merged = Dense(units=relu_size, activation='relu',
                             name='relu_dense')(x_merged)
            if add_dense:
                x_merged = Dense(
                    units=relu_size // 2, activation='relu', name='relu_dense_2')(x_merged)

            x_merged = Dense(units=output_size, activation='softmax',
                             name='softmax_dense')(x_merged)

    else:
        if (nb_passbands > 1):
            x_merged = Dense(units=relu_size, activation='relu',
                             name='relu_dense')(x_merged)
            # if add_dense:
            #    x_merged = Dense(units=relu_size//2, activation='relu',  name='relu_dense_2')(x_merged)

        x_merged = TimeDistributed(Dense(
            units=output_size, activation='linear'), name='time_dist'.format(j))(x_merged)

    # --------------- MODEL --------------- #
    input_layer = main_input_list + \
        [meta_input] if add_meta is not None else main_input_list
    output_layer = x_merged
    m_model = Model(input_layer, output_layer)

    param_str = 'dTCN_pb{}_n{}_drop{}_stack{}_dil{}_cv{}_cvW{}_out{}_outW{}'.format(nb_passbands,
                                                                                    sizenet, int(drop_frac * 100),
                                                                                    n_stacks, dilation_depth,
                                                                                    kernel_size, kernel_wavenet,
                                                                                    output_size, output_size_cw)
    if m_activation == 'wavenet':
        param_str += '_aW'
    if config_wavenet:
        param_str += '_cW'
    if causal:
        param_str += '_causal'

    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict


## ########################################################################## ##
# AUTO-ENCODERS
## ########################################################################## ##


## -------------------------------------------------------------------------- ##
# 4 - ED RNN
## -------------------------------------------------------------------------- ##
def net_ED_RNN_pb(input_dimension, sizenet,
                  embedding,
                  nb_passbands=1, max_lenght=None,
                  num_layers=1,
                  drop_frac=0.0,
                  model_type='LSTM',
                  bidirectional=True,
                  aux_in=True,
                  **kwargs):

    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}

    layer = dict_rnn[model_type]

    main_input_list = []  # aux_input_list=[]
    for j in range(nb_passbands):
        ndatapoints = max_lenght if isinstance(
            max_lenght, int) else max_lenght[j]

        main_input = Input(shape=(ndatapoints, input_dimension),
                           name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)

        # aux_input = Input(shape=(ndatapoints, input_dimension-1),
        #                   name='aux_input_pb{}'.format(j)) if aux_in else None
        # aux_input_list.append(aux_input)

    total_datapoints = np.sum(max_lenght)
    aux_input_concat = Input(shape=(
        total_datapoints, input_dimension - 1), name='aux_input_concat') if aux_in else None

    # --------------- ENCODER --------------- #
    encode_list = []

    for j in range(nb_passbands):
        encode = main_input_list[j]

        for i in range(num_layers):
            mcond = (i < num_layers - 1)
            wrapper = Bidirectional if bidirectional else lambda x: x
            encode = wrapper(layer(units=sizenet, return_sequences=mcond),
                             name='encode_pb{}_x{}'.format(j, i))(encode)

            ## Dropout ##
            if drop_frac > 0.0:
                encode = Dropout(
                    rate=drop_frac, name='encode_drop_pb{}_x{}'.format(j, i))(encode)

        ## Dense, Emb ##
        encode = Dense(activation='linear', name='encode_dense_pb{}'.format(
            j), units=embedding)(encode)
        encode_list.append(encode)

    ## Merge ##
    encode_merged = concatenate(
        encode_list, name='encode_concat') if nb_passbands > 1 else encode_list[0]
    if (nb_passbands > 1):
        encode_merged = Dense(units=embedding, activation='relu',
                              name='encode_relu_dense')(encode_merged)

    # --------------- DECODER --------------- #
    decode = RepeatVector(total_datapoints, name='repeat')(encode_merged)
    if aux_in:
        decode = concatenate([aux_input_concat, decode],
                             name='decode_aux_concat')

    for i in range(num_layers):
        ## Dropout ##
        if (drop_frac > 0.0) and (i > 0):  # skip for 1st layer (symmetry)
            decode = Dropout(rate=drop_frac,
                             name='drop_decode_x{}'.format(i))(decode)

        wrapper = Bidirectional if bidirectional else lambda x: x
        decode = wrapper(layer(units=sizenet,
                               return_sequences=True),
                         name='decode_x{}'.format(i))(decode)

    ## Output ##
    decode = TimeDistributed(
        Dense(units=1, activation='linear'), name='decode_time_dist')(decode)

    # --------------- MODEL --------------- #
    input_layer = main_input_list + \
        [aux_input_concat] if aux_in else main_input_list
    output_layer = decode
    m_model = Model(input_layer, output_layer)

    param_str = 'ED_{}_pb{}_n{}_x{}_drop{}_emb{}'.format(model_type,
                                                         nb_passbands, sizenet,
                                                         num_layers,
                                                         int(drop_frac * 100),
                                                         embedding)
    if bidirectional:
        param_str += '_bidir'

    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict


## -------------------------------------------------------------------------- ##
# 5 - ENCODER-DECODER TCN
## -------------------------------------------------------------------------- ##
def net_ED_tCNN_pb(input_dimension, sizenet,
                   kernel_size, embedding,
                   nb_passbands=1, max_lenght=None,
                   causal=False,
                   num_layers=3,
                   drop_frac=0.25,
                   m_reductionfactor=2,
                   m_activation='wavenet',
                   do_featurizer=True,
                   aux_in=True,
                   **kwargs):

    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}

    main_input_list = []
    for j in range(nb_passbands):
        if max_lenght is None:
            ndatapoints = None
        else:
            ndatapoints = max_lenght if isinstance(
                max_lenght, int) else max_lenght[j]
        main_input = Input(shape=(ndatapoints, input_dimension),
                           name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)

    total_datapoints = np.sum(max_lenght)
    aux_input_concat = Input(shape=(
        total_datapoints, input_dimension - 1), name='aux_input_concat') if aux_in else None

    m_padding = 'causal' if causal else 'same'

    # --------------- ENCODER --------------- #
    encode_list = []
    ntimes_list = []

    for j in range(nb_passbands):
        encode = main_input_list[j]

        for i in range(num_layers):
            if isinstance(sizenet, int):
                size_n = sizenet
            else:
                size_n = sizenet[i] if len(sizenet) == num_layers else sizenet

            ## Convo ##
            encode = Conv1D(filters=size_n,
                            kernel_size=kernel_size,
                            padding=m_padding,
                            name='encode_pb{}_x{}_C{}_N{}'.format(j, i, kernel_size, size_n))(encode)

            ## Activation ##
            if m_activation == 'wavenet':
                encode = keras.layers.multiply([Activation('tanh', name='encode_pb{}_x{}_activation_tanh'.format(j, i))(encode),
                                                Activation('sigmoid', name='encode_pb{}_x{}_activation_sigmoid'.format(j, i))(encode)],
                                               name='encode_pb{}_x{}_activation_multiply'.format(j, i))
            else:
                encode = Activation(m_activation, name='encode_pb{}_x{}_activation_{}'.format(
                    j, i, m_activation))(encode)

            ## Poling ##
            if m_reductionfactor > 0:
                encode = MaxPooling1D(pool_size=m_reductionfactor,
                                      name='encode_pb{}_x{}_maxpool{}'.format(j, i, m_reductionfactor))(encode)

            ## Dropout ##
            if drop_frac > 0:
                encode = SpatialDropout1D(drop_frac,
                                          name='encode_pb{}_x{}_sdrop{}'.format(j, i, int(drop_frac * 100)))(encode)

        ## Dense, Emb ##
        encode = Dense(units=embedding, name='encode_pb{}_x{}_dense'.format(
            j, i), activation='relu')(encode)
        ntimes = encode.get_shape().as_list()[1]  # max_lenght
        ntimes_list.append(ntimes)
        if do_featurizer:
            encode = Lambda(
                lambda y: y[:, -1, :], name='encode_pb{}_embedding'.format(j))(encode)  # embedding

        ## Store ##
        encode_list.append(encode)

    ## Merge ##
    encode_merged = concatenate(
        encode_list, name='encode_concat') if nb_passbands > 1 else encode_list[0]
    if (nb_passbands > 1):
        encode_merged = Dense(units=embedding, activation='relu',
                              name='encode_relu_dense')(encode_merged)

    # --------------- DECODER --------------- #
    decode = encode_merged
    if do_featurizer:
        decode = RepeatVector(np.sum(ntimes_list))(decode)

    if aux_in & (np.sum(ntimes_list) == total_datapoints):
        decode = concatenate([aux_input_concat, decode],
                             name='decode_aux_concat')

    for i in range(num_layers):
        if isinstance(sizenet, int):
            size_n = sizenet
        else:
            size_n = sizenet[-i - 1] if (len(sizenet)
                                         == num_layers) else sizenet

        ## Upsample ##
        if m_reductionfactor > 0:
            decode = UpSampling1D(size=m_reductionfactor,
                                  name='decode_x{}_upsample{}'.format(i, m_reductionfactor))(decode)

        ## Convo ##
        decode = Conv1D(filters=size_n,
                        kernel_size=kernel_size,
                        padding=m_padding,
                        name='decode_x{}_cv{}_n{}'.format(i, kernel_size, size_n))(decode)

        ## Activation ##
        if m_activation == 'wavenet':
            decode = keras.layers.multiply([Activation('tanh', name='decode_x{}_activation_tanh'.format(i))(decode),
                                            Activation('sigmoid', name='decode_x{}_activation_sigmoid'.format(i))(decode)],
                                           name='decode_pb{}_x{}_activation_multiply'.format(j, i))
        else:
            decode = Activation(
                m_activation, name='decode_x{}_activation_{}'.format(i, m_activation))(decode)

        ## Dropout ##
        if drop_frac > 0:
            decode = SpatialDropout1D(
                drop_frac, name='decode_x{}_sdrop_{}'.format(i, int(drop_frac * 100)))(decode)

    ## Output ##
    decode = TimeDistributed(
        Dense(units=1, activation='softmax', name='decode_time_dist'))(decode)

    # --------------- MODEL --------------- #
    input_layer = main_input_list + \
        [aux_input_concat] if aux_in else main_input_list
    output_layer = decode
    m_model = Model(input_layer, output_layer)

    param_str = 'ED_tCNN_pb{}_n{}_x{}_drop{}_cv{}'.format(nb_passbands, sizenet, num_layers,
                                                          int(drop_frac * 100), kernel_size)
    if do_featurizer:
        param_str += '_emb{}'.format(embedding)
    if m_activation == 'wavenet':
        param_str += '_aW'
    if causal:
        param_str += '_causal'

    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict


## -------------------------------------------------------------------------- ##
# 6 - ENCODER-DECODER DILATED-TCN
## -------------------------------------------------------------------------- ##
def net_ED_dTCN_pb(input_dimension, sizenet,
                   embedding,
                   n_stacks, list_dilations=None, max_dilation=None,
                   nb_passbands=1, max_lenght=None,
                   causal=False,
                   drop_frac=0.25,
                   config_wavenet=True,
                   m_activation='wavenet',
                   use_skip_connections=True,
                   kernel_size=3, kernel_wavenet=1,
                   aux_in=True,
                   **kwargs):

    if list_dilations is None:
        list_dilations = list(range(max_dilation))

    param_dict = {key: value for key, value in locals().items()}
    param_dict.update(kwargs)
    param_dict = {k: v for k, v in param_dict.items()
                  if k not in ['kwargs']
                  and not isinstance(v, types.FunctionType)}
    for key, value in param_dict.items():
        if isinstance(value, np.int64):
            param_dict[key] = int(value)

    dilation_depth = len(list_dilations)
    m_dilations = [2**i for i in list_dilations]

    main_input_list = []  # aux_input_list=[]
    for j in range(nb_passbands):
        ndatapoints = max_lenght if isinstance(
            max_lenght, int) else max_lenght[j]
        #aux_input  = Input(shape = (max_lenght, input_dimension-1), name = 'aux_input') if aux_in else None
        main_input = Input(shape=(ndatapoints, input_dimension),
                           name='main_input_pb{}'.format(j))
        main_input_list.append(main_input)

    total_datapoints = np.sum(max_lenght)
    aux_input_concat = Input(shape=(
        total_datapoints, input_dimension - 1), name='aux_input_concat') if aux_in else None

    m_padding = 'causal' if causal else 'same'

    # --------------- ENCODER --------------- #
    encode_list = []
    for j in range(nb_passbands):
        encode = main_input_list[j]

        ## Conv ##
        encode = Conv1D(filters=sizenet,
                        kernel_size=kernel_size,
                        padding=m_padding,
                        name='encode_pb{}_initconv'.format(j))(encode)

        ## Residuals, Skip_connections ##
        skip_connections = []
        for m_stack in range(n_stacks):
            for m_dilation_rate in m_dilations:
                encode, skip_encode = residual_block(encode, sizenet,
                                                     m_dilation_rate, m_stack,
                                                     causal,
                                                     m_activation, drop_frac,
                                                     kernel_size, name='encode_pb{}_'.format(j))
                skip_connections.append(skip_encode)

        if use_skip_connections:
            encode = keras.layers.add(
                skip_connections, name='encode_pb{}_add_skip_connections'.format(j))

        encode = Activation('relu', name='encode_pb{}_relu'.format(j))(encode)

        ## Wavenet config ##
        if config_wavenet:
            encode = Conv1D(filters=sizenet,
                            kernel_size=kernel_wavenet,
                            padding='same',
                            name='encode_pb{}_n{}_cv{}_cw'.format(j, sizenet,
                                                                  kernel_wavenet))(encode)
            #
            encode = Activation(
                'relu', name='encode_pb{}_relu_cw'.format(j))(encode)
            #
            encode = Conv1D(filters=embedding,
                            kernel_size=kernel_wavenet,
                            padding='same',
                            name='encode_pb{}_n{}_cv{}_cw'.format(j, embedding,
                                                                  kernel_wavenet))(encode)
            #
            encode = Activation(
                'softmax', name='encode_pb{}_softmax_cw'.format(j))(encode)
        else:
            encode = Dense(units=embedding, activation='relu',
                           name='encode_pb{}_dense'.format(j))(encode)

        ## Emb ##
        encode = Lambda(
            lambda y: y[:, -1, :], name='encode_pb{}_embedding'.format(j))(encode)  # embedding

        # Store
        encode_list.append(encode)

    ## Merge ##
    encode_merged = concatenate(
        encode_list, name='encode_concat') if nb_passbands > 1 else encode_list[0]
    if (nb_passbands > 1):
        encode_merged = Dense(units=embedding, activation='relu',
                              name='encode_relu_dense')(encode_merged)

    # --------------- DECODER --------------- #
    decode = encode_merged
    decode = RepeatVector(total_datapoints)(decode)
    if aux_in:
        decode = concatenate([aux_input_concat, decode],
                             name='decode_aux_concat')

    ## Conv ##
    decode = Conv1D(filters=sizenet,
                    kernel_size=kernel_size,
                    padding=m_padding,
                    name='decode_initconv')(decode)

    ## Residuals, Skip_connections ##
    skip_connections = []
    for m_stack in range(n_stacks):
        for m_dilation_rate in m_dilations:
            decode, skip_decode = residual_block(decode, sizenet,
                                                 m_dilation_rate, m_stack,
                                                 causal,
                                                 m_activation, drop_frac,
                                                 kernel_size, name='decode_')
            skip_connections.append(skip_decode)

    if use_skip_connections:
        decode = keras.layers.add(
            skip_connections, name='decode_add_skip_connections')

    decode = Activation('relu', name='decode_relu')(decode)

    ## Wavenet config ##
    if config_wavenet:
        decode = Conv1D(filters=sizenet,
                        kernel_size=kernel_wavenet,
                        padding='same',
                        name='decode_n{}_cv{}_cw'.format(sizenet, kernel_wavenet))(decode)
        #
        decode = Activation('relu', name='decode_relu_cw')(decode)
        #
        decode = Conv1D(filters=1,
                        kernel_size=kernel_wavenet,
                        padding='same',
                        name='decode_n{}_cv{}_cw'.format(1, kernel_wavenet))(decode)
        #
        decode = TimeDistributed(Activation(
            'softmax', name='decode_softmax'), name='time_dist')(decode)
    else:
        decode = TimeDistributed(
            Dense(units=1, activation='relu'), name='time_dist')(decode)

    # --------------- MODEL --------------- #
    input_layer = main_input_list + \
        [aux_input_concat] if aux_in else main_input_list
    #input_layer  = [main_input, aux_input] if aux_in else main_input
    output_layer = decode
    m_model = Model(input_layer, output_layer)

    param_str = 'ED_dTCN_pb{}_n{}_drop{}_stack{}_dil{}_cv{}_cvW{}_emb{}'.format(nb_passbands, sizenet,
                                                                                int(drop_frac * 100),
                                                                                n_stacks, dilation_depth,
                                                                                kernel_size, kernel_wavenet,
                                                                                embedding)
    if m_activation == 'wavenet':
        param_str += '_aW'
    if config_wavenet:
        param_str += '_cW'
    if causal:
        param_str += '_causal'

    print('>>>> param_str = ', param_str)
    return m_model, param_str, param_dict
