import sys
import numpy as np
import argparse
import math

import keras
from keras import backend as K
from keras import initializers
from keras import optimizers
from keras import losses
from keras.models import load_model, model_from_json
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, MaxPooling1D, UpSampling1D, Activation, Dropout
from keras.layers import Flatten, BatchNormalization, GlobalMaxPooling1D
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def main(arguments):

    name = arguments.name

    in_len = arguments.in_len
	
    print('Make x,y file')
    train = 'train.txt'
    x_train, y_train = make_x_y(train)
    print('Start Training')
    train_net(x_train, y_train, name, in_len)

    print('Make x,y file')
    test = 'test.txt'
    x_test, y_test = make_x_y(test)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print('Evaluate')
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(name + '.h5')
    loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', f1_m, matthews_correlation])
    loaded_model.summary()
	
    predictions = np.argmax(loaded_model.predict(x_test),axis=1)
    correct = 0
    y_list = list()
    for pred, y in zip(predictions, y_test):
        if y[0] == 0.:
            y = 1
        elif y[0] == 1.:
            y = 0
        else:
            print('a')
        if pred == y:
            correct += 1
        y_list.append(y)
    tn, fp, fn, tp = confusion_matrix(y_list, predictions).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    prec = (tp)/(tp+fp)
    rec = (tp)/(tp+fn)
    f1 = 2*((prec*rec)/(prec+rec))
    mcc = ((tp*tn)-(fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    spec = (tn)/(tn+fp)
    print('Accuracy = %f' % (acc))
    print('F1 = %f' % (f1))
    print('MCC = %f' % (mcc))
    print('Specificity = %f' % (spec))
	
    with open(name + '.txt', 'w') as f:
        f.write('Accuracy = %f\n' % (acc))
        f.write('F1 = %f\n' % (f1))        
        f.write('MCC = %f\n' % (mcc))
        f.write('Specificity = %f\n' % (spec))


def make_x_y(path):
    x = list()
    y = list()
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                label = line.split('|')[1]
                if label == '1':
                    y.append([0., 1.])
                else:
                    y.append([1., 0.])
            else:
                seq = line[:-1]
                x.append(make_one_hot(seq))
    return x, y


def make_one_hot(seq):
    result = list()
    for nt in seq:
        if nt == 'A':
            result.append([1., 0., 0., 0., 0.])
        elif nt == 'U':  # U/T = 0100 T = DNA
            result.append([0., 1., 0., 0., 0.])
        elif nt == 'T':
            result.append([0., 1., 0., 0., 0.])
        elif nt == 'G':
            result.append([0., 0., 1., 0., 0.])
        elif nt == 'C':
            result.append([0., 0., 0., 1., 0.])
        elif nt == 'N':
            result.append([0., 0., 0., 0., 1.])
        elif nt == 'Z':
            result.append([0., 0., 0., 0., 0.])
        elif nt == '\n':
            continue
        else:
            print(nt)
            print('error one hot')
            sys.exit()
    return result


def train_net(x_train, y_train, name, max_len):

    x_train = np.array(x_train)
    y_train = np.array(y_train)
	
    input_shape = Input(shape=(max_len, 5))

    autoencoder = Model(input_shape, decoder(encoder(input_shape)))
    autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')
    print(autoencoder.summary())
	
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    mcp_save = ModelCheckpoint(name + '_ae.h5', save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True, verbose=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=128,
                    validation_split=0.1,
                    callbacks=[mcp_save, reduce_lr_loss, earlyStopping],
                    shuffle=True)

    autoencoder_json = autoencoder.to_json()
    with open(name + '_ae.json', 'w') as json_file:
        json_file.write(autoencoder_json)
	
    final_model = Model(input_shape, classifier(encoder(input_shape)))
	
    for l1,l2 in zip(final_model.layers[0:5],autoencoder.layers[0:5]):
        l1.set_weights(l2.get_weights())
		
	
    for layer in final_model.layers[0:5]:
        layer.trainable = True
	
    final_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    print(final_model.summary())
	
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    mcp_save = ModelCheckpoint(name + '.h5', save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True, verbose=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

	
    final_model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=128,
                    validation_split=0.1,
                    callbacks=[mcp_save, reduce_lr_loss, earlyStopping],
                    shuffle=True)
					
    final_model_json = final_model.to_json()
    with open(name + '.json', 'w') as json_file:
        json_file.write(final_model_json)
	
def encoder(input_seq):
    
    x = Conv1D(filters=128, kernel_size=12, strides=1, padding='same', activation='relu')(input_seq)
    x = MaxPooling1D(2)(x)
    x = Conv1D(filters=128, kernel_size=6, strides=1, padding='same', activation='relu')(x)
    encoded = MaxPooling1D(2)(x)
    return encoded

def decoder(encoded_seq):
    x = Conv1D(filters=128, kernel_size=6, activation='relu', padding='same')(encoded_seq)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=128, kernel_size=12, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(5, kernel_size=1, activation='sigmoid', padding='same')(x)
    return decoded
	
def classifier(encoded_seq):
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu')(encoded_seq)
    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    #x = Dropout(0.5)(x)
    x = Flatten()(x)
    out = Dense(2, activation='softmax')(x)
    return out


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def parse_arguments(parser):
    parser.add_argument('--name', type=str,
                        help='Name of the network. Will be used as the filename to save results', required=True)

    parser.add_argument('--in_len', type=int, help='Length of input sequence', required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    #python ae.py --name y --in_len 200 

    parser = argparse.ArgumentParser(
        description='circular RNA classification from other long non-coding RNA using deep learning')

    args = parse_arguments(parser)

    main(args)
