from tensorflow.keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from tensorflow import constant

############### Model evaluation ###############
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


############### Model configuration ###############
def name_model(algorithm_type = '', params = {}):
    name = algorithm_type
    for key, value in params.items():
        if key == 'optimizer':
            if type(value) == keras.optimizers.optimizer_v2.adam.Adam:
                name += f"_adam"
            else:
                name += f"_else"
        elif key == 'regularizer':
            name += f"_{value['input']}_{value['hidden']}_{value['bias']}"
            
        else:
            name += f"_{value}"
        
    return name

def get_optimizer(optimizer_type, learning_rate):
    if optimizer_type == 'Adam':
        return Adam(learning_rate = learning_rate, beta_1=0.9, beta_2=0.999)
    else:
        print("None of optimizers")

from keras import backend as K
from keras.regularizers import Regularizer, l1, l2
import numpy as np

class SGL21(Regularizer):
    def __init__(self, l_value=0.0001):
        self.l_value = K.cast_to_floatx(l_value)

    def __call__(self, x):
        const_coeff = np.sqrt(K.int_shape(x)[1])
        return self.l_value*( const_coeff*K.sum(K.sqrt(K.sum(K.square(x), axis=1))) + K.sum(K.abs(x)) )

    def get_config(self):
        return {'l_value': float(self.l_value)}

def get_regularizer(type_coeff):
    if type_coeff:
        reg_type  = type_coeff.split('_')[0]
        l_value = float(type_coeff.split('_')[1])
        if reg_type == 'L1': # Lasso regularization
            return l1(l_value)
        elif reg_type == 'L2': # Ridge regularization
            return l2(l_value)
        elif reg_type == 'L21': # Group Lasso regularization
            return SGL21(l_value)
        else:
            return 'input valid regularizer reg_type and l_value'
    else:
        return None
        

default_params = {
    'rnn_layers'     : 1,
    'rnn_neurons'    : 64,
    'dnn_layers'     : 3,
    'dnn_neurons'    : 64,
    'activation'     : 'softmax',
    'loss'           : 'categorical_crossentropy',
    'metrics'        : 'accuracy',
    'optimizer_type' : 'Adam',
    'learning_rate'  : 0.001,
    'regularizer'    : {'input': None, 'hidden': None, 'bias': None}
}

def MLP_CLS(x_dim, y_dim, params):
    dnn_layers         = params.get('dnn_layers',     default_params['dnn_layers'])
    dnn_neurons        = params.get('dnn_neurons',    default_params['dnn_neurons'])
    activation         = params.get('activation',     default_params['activation'])
    loss               = params.get('loss',           default_params['loss'])
    metrics            = params.get('metrics',        default_params['metrics'])
    optimizer_type     = params.get('optimizer_type', default_params['optimizer_type'])
    learning_rate      = params.get('learning_rate',  default_params['learning_rate'])
    regularizer_input  = get_regularizer(params.get('regularizer',    None).get('input', None))
    regularizer_hidden = get_regularizer(params.get('regularizer',    None).get('hidden', None))
    regularizer_bias   = get_regularizer(params.get('regularizer',    None).get('bias', None))
    
    model_input  = Input(shape=(x_dim,), name='model_input')
    dense_output = Dense(dnn_neurons, kernel_regularizer=regularizer_input, bias_regularizer=regularizer_bias, 
                         name="input_layer")(model_input)

    for i in range(dnn_layers-1):
        dense_output = Dense(dnn_neurons, kernel_regularizer=regularizer_hidden, bias_regularizer=regularizer_bias, 
                             name=f"hidden_{i+1}")(dense_output)

    model_output = Dense(y_dim, kernel_regularizer=regularizer_hidden, bias_regularizer=regularizer_bias,
                         name=f"model_output", activation=activation)(dense_output)  
    model = Model(model_input, model_output)
    
    model.compile(loss = loss, optimizer = get_optimizer(optimizer_type, learning_rate), metrics = metrics)
    
    return model

def LSTM_CLS(x_len, x_dim, y_dim, params):
    rnn_layers         = params.get('rnn_layers',     default_params['rnn_layers'])
    rnn_neurons        = params.get('rnn_neurons',    default_params['rnn_neurons'])
    
    dnn_layers         = params.get('dnn_layers',     default_params['dnn_layers'])
    dnn_neurons        = params.get('dnn_neurons',    default_params['dnn_neurons'])
    activation         = params.get('activation',     default_params['activation'])
    loss               = params.get('loss',           default_params['loss'])
    metrics            = params.get('metrics',        default_params['metrics'])
    optimizer_type     = params.get('optimizer_type', default_params['optimizer_type'])
    learning_rate      = params.get('learning_rate',  default_params['learning_rate'])
    regularizer_input  = params.get('regularizer',    None).get('input', None)
    if regularizer_input:
        reg_type  = regularizer_input.split('_')[0]
        reg_coeff = float(regularizer_input.split('_')[1])
    regularizer_hidden = params.get('regularizer',    None).get('hidden', None)
    regularizer_bias   = params.get('regularizer',    None).get('bias', None)
    
    model_input  = Input(shape=(x_len, x_dim), name='model_input')
    
        # encoder module
    if rnn_layers == 1:
        rnn_output, state_h, state_c = LSTM(rnn_neurons, kernel_regularizer=regularizer_input, bias_regularizer=regularizer_bias, 
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
    
    model.compile(loss = loss, optimizer = get_optimizer(optimizer_type, learning_rate), metrics = metrics)
    
    return model
