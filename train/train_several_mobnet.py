import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import tensorflow as tf
import keras
import numpy as np
import tensorflow
from keras.optimizers import SGD

import sys
sys.path.append("..")
from utils import *
from funcs import *

from mobnet import model_mobnet
from training_functions import *

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

data_path = 'data_2020/'
train_csv = data_path + 'evaluation_setup/fold1_train.csv'
val_csv = data_path + 'evaluation_setup/fold1_evaluate.csv'
feat_path = 'features/logmel128_scaled_d_dd/'
experiments = 'exp_mobnet'

if not os.path.exists(experiments):
    os.makedirs(experiments)


params = [[0.1, 200, 0.4], [0.1, 100, 0.1], [0.05, 150, 0.4], [0.1, 300, 0.4]]
no = 3
for param in params:
    no += 1
    print('---------------------- NO: {} params: {} -------------------------------'.format(no, param))
    # random sample data, to keep all three classes have similar number of training samples
    total_csv = balance_class_data(train_csv, experiments)

    num_audio_channels = 2
    num_freq_bin = 128
    num_time_bin = 461
    num_classes = 3
    max_lr = param[0]
    batch_size = 32
    num_epochs = param[1]
    mixup_alpha = param[2]
    sample_num = len(open(train_csv, 'r').readlines()) - 1
    

    data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
    y_val = keras.utils.to_categorical(y_val, num_classes)

    model = model_mobnet(num_classes, input_shape=[num_freq_bin, num_time_bin, 3*num_audio_channels], num_filters=24, wd=1e-3)

    model.compile(loss='categorical_crossentropy',
                  optimizer = SGD(lr=max_lr,decay=0, momentum=0.9, nesterov=False),
                  metrics=['accuracy']) #ori

    lr_scheduler = LR_WarmRestart(nbatch=np.ceil(sample_num/batch_size), Tmult=2,
                                  initial_lr=max_lr, min_lr=max_lr*1e-4,
                                  epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0]) 
    save_path = experiments + "/model-mixup_alpha:{}-max_lr:{}-total_epochs:{}-no:{}.hdf5".format(mixup_alpha, max_lr,               num_epochs, no)
    checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    callbacks = [lr_scheduler, checkpoint]

    train_data_generator = Generator_balanceclass_timefreqmask_nocropping_splitted(feat_path, train_csv, total_csv, experiments,     num_freq_bin, 
                                  batch_size=batch_size,
                                  alpha=mixup_alpha, splitted_num=4)()

    history = model.fit_generator(train_data_generator,
                                  validation_data=(data_val, y_val),
                                  epochs=num_epochs, 
                                  verbose=1, 
                                  workers=4,
                                  max_queue_size = 100,
                                  callbacks=callbacks,
                                  steps_per_epoch=1
                                  ) 
