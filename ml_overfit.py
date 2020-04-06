import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.preprocessing
from PIL import Image

import argparse

"""

Connections:

    All: PVTA
    Time

    Input -> FC1, Conv1, FC2
    Time  -> FC1, FC2
    Conv1 -> FC1
    FC1   -> FC2, Conv2
    Conv2 -> FC2
    FC2   -> Output

"""


folder = 'NpArrays/'


def _load_data():
    print ('Loading')
    #T*X*Y
    # p_in = np.load(folder+'Pressure.npy')
    # p_in = None
    #T*X*Y
    t_in = np.load(folder+'Temperature.npy')

    d_t = t_in - np.roll(t_in,1, axis=0)
    d_t[0]=0

    # print ('--t--')
    # print (t_in[:,20,20])

    #T*X*Y*2
    v_in = np.load(folder+'Velocity.npy')
    # v_in[0]=0
    #T*X*Y*2
    #Accel based on previous step
    a_in = v_in - np.roll(v_in,1,axis=0)
    a_in[0]=0
    #T
    time = np.load(folder+'Time.npy')

    print ('loaded shape=',v_in.shape)

    return t_in, d_t, v_in, a_in, time


def _concat_data():
    t,d_t,v,a,time = _load_data()
    return np.concatenate((t, d_t, v, a),axis=3), time
    # return np.concatenate((p,t,v,a),axis=3), time

def load_and_norm_data():
    batch, time = _concat_data()

    eps = 1e-6

    #Normalize by channel
    batch -= batch.mean(axis=3)[:,:,:,None]
    batch /= np.sqrt((batch**2).mean(axis=3))[:,:,:,None] + .001

    # print(batch[:,20,20])

    x_amt = batch.shape[1]
    y_amt = batch.shape[2]
    return batch, x_amt,y_amt, time

def get_num_channels():
    #Number of inputs with 1 channel
    c1_amt = 2
    #Number of inputs with 2 channels
    c2_amt = 2
    #Number of inputs
    var_ct = c1_amt + c2_amt
    #Total number of channels
    num_channels = c1_amt*1 + c2_amt*2
    return num_channels

def create_model(x_amt, y_amt, num_channels):
    from edge_initializer import EdgeInitializer

    pix_ct = int(x_amt * y_amt)

    num_nodes = int(num_channels * pix_ct)

    num_filters = num_channels * 6
    kernel_size = 4

    X = tf.keras.Input(shape=(x_amt,y_amt,num_channels))

    padl = int(kernel_size -1)
    padl = [(0,0),(padl,padl),(padl,padl),(0,0)]

    def pad(x): return tf.pad(x, padl, 'symmetric')

    ei = EdgeInitializer(sep_channels=True,extra_dims=None)

    conv1 = pad(X)
    conv1 = tf.keras.layers.Conv2D(num_filters, kernel_size, activation=tf.nn.sigmoid, kernel_initializer=ei, padding='valid', name='conv1_1')(X)
    conv1 = pad(conv1)
    conv1 = tf.keras.layers.Conv2D(num_filters, kernel_size, activation=tf.nn.sigmoid, kernel_initializer=ei, padding='valid', name='conv1_2')(conv1)
    # conv1 = tf.keras.layers.Conv2D(num_filters, kernel_size, activation=tf.nn.sigmoid, padding='same', name='conv1_3')(conv1)

    conv2 = pad(X)
    conv2 = tf.keras.layers.Conv2D(num_filters, kernel_size, activation=tf.nn.sigmoid, kernel_initializer=ei, padding='valid', name='conv2_1')(X)
    conv2 = pad(conv2)
    conv2 = tf.keras.layers.Conv2D(num_filters, kernel_size, activation=tf.nn.sigmoid, kernel_initializer=ei, padding='valid', name='conv2_2')(conv2)


    def fc_initializer(shape,dtype=None):
        """ Requires X first in appends list
        """
        #Use random normal / 100 (to allow start of gradient descent while allowing input X to pass through unimpeded)
        init = tf.keras.initializers.glorot_normal()(shape,dtype)
        init /= 100
        #Add .5 to self-connections in input connections
        #To allow x->x to appear almost immediately
        init+= tf.keras.initializers.identity(gain=1)(shape[2:],dtype)[None,None]
        print ('fc_initializer:',init)
        return init

    #Fully connected as 1x1 Conv2D
    fc1_input = tf.concat([X,conv1],axis=3)
    fc1 = tf.keras.layers.Conv2D(num_filters, 1, kernel_initializer=fc_initializer, bias_initializer='zeros', use_bias=True,
            activation=tf.nn.sigmoid, padding='same',name='fc1')(fc1_input)


    fc2_input = tf.concat((fc1,conv2),axis=3)
    fc2 = tf.keras.layers.Conv2D(num_filters, 1, kernel_initializer=fc_initializer, bias_initializer='zeros', use_bias=True,
                activation=tf.nn.sigmoid,padding='same',name='fc2')(fc2_input)

    # out_input = tf.concat((fc2),axis=3)
    out_input=fc2
    out = tf.keras.layers.Conv2D(num_channels, 1, kernel_initializer=fc_initializer, bias_initializer='zeros', use_bias=True,
                activation=tf.nn.sigmoid,padding='same',name='out')(out_input)

    #Reverse the sigmoid function
    #Consider scaling it
    # ln (y / (1-y) )
    out = -tf.math.log(1/(out+.00001) - 1)

    model = tf.keras.Model(inputs=X, outputs=out)

    return model

def sigmoid(x):
    return 1/(1+np.exp(-x))

TRAIN_REGION = None

def print_tensor(tensor,message=''):
    print (tensor)
    return tf.keras.backend.print_tensor(tensor,message)

def train(epochs,resume_training):

    batch_x, x_amt,y_amt, time_scale = load_and_norm_data()

    print ('batch values')
    print (batch_x.mean(axis=3),'\n',np.linalg.norm(batch_x,axis=3))
    print (batch_x.min(axis=3),'\n',batch_x.max(axis=3))

    num_channels = get_num_channels()

    model = create_model(x_amt, y_amt, num_channels)

    # channel_distinctions = [0, 1, (2,3), (4,5)]
    pix_ct = x_amt*y_amt


    print ('model = ',model)
    # Euclidean Norm, ord=2      Sqrt(mean(Y**2))+.001
    # y_norm = (np.abs(batch_x).mean(axis=(0,1,2))) + 1e-4
    channel_value = np.array([0, 1, 0, 0, .7, .7])
    channel_power = np.array([1, 1, 1, 1,  1,  1])
    channel_value = channel_value[None,None,None,:]
    def accuracy(Y, X):
        # (X-Y)^2 / magn(Y,axis=Channels)^2
        acc = tf.reduce_mean(
                tf.math.pow(tf.math.abs(
                    tf.math.subtract(X,Y)),
                    channel_power)
                * channel_value )
        # acc = print_tensor(acc,'\nAccuracy')
        return acc


    # y_ph = tf.placeholder(tf.float32, [None,x_amt,y_amt,num_channels], name='labels')
    log_dir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5)

    # loss = accuracy(model,y_ph)
    learning_rate = .001

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,rho=.9)
    model.compile(optimizer=optimizer, loss=accuracy)

    if resume_training:
        print ('Loading weights')
        model.load_weights('weights')

    if epochs is None:
        epochs = 50
    batch_size = 4

    batch_y = np.roll(batch_x, -1, axis=0)

    batch_x = batch_x[:TRAIN_REGION]
    batch_y = batch_y[:TRAIN_REGION]

    # print (batch_y)
    # exit()

    save_rate = 100
    epoch_epoch = int(np.ceil(epochs/save_rate))

    for e in range(epoch_epoch):
        done=False
        ep = epochs - e * save_rate
        ep = min(save_rate,ep)
        if epoch_epoch > 1:
            print ('\n\n----- Greater Epoch ',e+1,'/',epoch_epoch,'-------------\n\n')
        try:
            history = model.fit (batch_x,batch_y,batch_size=batch_size,epochs=ep, callbacks=[tensorboard_callback])
        except (KeyboardInterrupt):
            print ("\nInterrupt - saving weights")
            model.save_weights('weights')
            exit(0)

        #If exited early (in which case you don't have history)
        #   or you have history and loss is not Nan
        if np.isfinite(history.history['loss']).all():
           #Save weights
            model.save_weights('weights')
        else:
            #Else loss is nan and exit loop
            print ('Nan loss: exiting')
            print (history)
            breakpt

    print ('Done')

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--cont',action='store_true')
    parser.add_argument('--train_region',type=int,default=None)
    args = parser.parse_args()

    if args.train_region is not None:
        TRAIN_REGION=args.train_region

    train(args.epochs,args.cont)
#
