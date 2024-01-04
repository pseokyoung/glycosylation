from tensorflow.keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from tensorflow import constant

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def class_scores(y_real, y_pred, rounding=4, average=None):
    accuracy  = 100*np.array(accuracy_score(y_real, y_pred)).round(rounding)
    precision = 100*np.array(precision_score(y_real, y_pred, average=average)).round(rounding)
    recall    = 100*np.array(recall_score(y_real, y_pred, average=average)).round(rounding)
    f1        = 100*np.array(f1_score(y_real, y_pred, average=average)).round(rounding)

    return accuracy, precision, recall, f1

def metrics_classification(model, test_x, test_y,
                           rounding=4, average=None):
    test_loss  = model.evaluate(test_x, test_y, verbose=0)[0]
    prediction = model.predict(test_x, verbose=0).round(0).astype(int)
    accuracy, precision, recall, f1 = class_scores(test_y, prediction,
                                                   rounding=rounding, average=average)

    return test_loss, accuracy, precision, recall, f1

def MLP(x_dim, y_dim, params):
    dnn_layers  = params['dnn_layers']
    dnn_neurons = params['dnn_neurons']
    activation  = params['activation']
    loss        = params['loss']
    metrics     = params['metrics']
    optimizer   = params['optimizer']
    regularizer_input  = params['regularizer']['input']
    regularizer_hidden = params['regularizer']['hidden']
    regularizer_bias   = params['regularizer']['bias']
    
    model_input  = Input(shape=(x_dim,), name='model_input')
    dense_output = Dense(dnn_neurons, kernel_regularizer=regularizer_input, bias_regularizer=regularizer_bias, 
                         name="input_layer")(model_input)

    for i in range(dnn_layers-1):
        dense_output = Dense(dnn_neurons, kernel_regularizer=regularizer_hidden, bias_regularizer=regularizer_bias, 
                             name=f"hidden_{i+1}")(dense_output)

    model_output = Dense(y_dim, kernel_regularizer=regularizer_hidden, bias_regularizer=regularizer_bias,
                         name=f"model_output", activation=activation)(dense_output)  
    model = Model(model_input, model_output)
    
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    
    return model

def LSTM(x_len, x_dim, y_dim, params):
    rnn_layers  = params['rnn_layers']
    rnn_neurons = params['rnn_neurons']
    dnn_layers  = params['dnn_layers']
    dnn_neurons = params['dnn_neurons']
    activation  = params['activation']
    loss        = params['loss']
    metrics     = params['metrics']
    optimizer   = params['optimizer']
    regularizer_input  = params['regularizer']['input']
    regularizer_hidden = params['regularizer']['hidden']
    regularizer_bias   = params['regularizer']['bias']
    
    model_input  = Input(shape=(x_len, x_dim), name='model_input')
    
        # encoder module
    if rnn_layers == 1:
        rnn_output, state_h, state_c = LSTM(params['rnn_neurons'], kernel_regularizer=regularizer_input, bias_regularizer=regularizer_bias, 
                                            return_state=True, name='rnn_1')(model_input)
        # encoder_states = [state_h, state_c]

    else:
        for i in range(rnn_layers):
            #first encoder layer
            if i==0: 
                rnn_output = LSTM(rnn_neurons, kernel_regularizer=regularizer_input, bias_regularizer=regularizer_bias, 
                                  return_sequences=True, name="encoder_1")(model_input)
            #mediate encoder layer
            elif i < rnn_layers-1: 
                rnn_output = LSTM(rnn_neurons, kernel_regularizer=regularizer_hidden, bias_regularizer=regularizer_bias, 
                                  return_sequences=True, name=f"encoder_{i+1}")(rnn_output)
            #last encoder layer
            else: 
                rnn_output, state_h, state_c  = LSTM(rnn_neurons, kernel_regularizer=regularizer_hidden, 
                                                     return_state=True, name=f"encoder_{i+1}")(rnn_output)
                # encoder_states = [state_h, state_c]

    # dense module
    if dnn_layers == 1:
        dnn_output = Dense(dnn_neurons, kernel_regularizer=regularizer_hidden, bias_regularizer=regularizer_bias, 
                           name='dense_1')(rnn_output)
    else:
        for i in range(dnn_layers):
            #first dense layer
            if i==0:
                dnn_output = Dense(dnn_neurons, kernel_regularizer=regularizer_hidden, bias_regularizer=regularizer_bias, 
                                   name='dense_1')(rnn_output)
            #mediate encoder layer
            else:
                dnn_output = Dense(dnn_neurons, kernel_regularizer=regularizer_hidden, bias_regularizer=regularizer_bias, 
                                   name=f'dense_{i+1}')(dnn_output)
    
    model_output = Dense(y_dim, kernel_regularizer=regularizer_hidden, bias_regularizer=regularizer_bias, activation=activation, 
                         name=f'model_output')(dnn_output)
    
    model = Model(model_input, model_output)
    
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    
    return model
