import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import ml_overfit
import argparse 
import time

import sys


parser = argparse.ArgumentParser()

parser.add_argument('--clip',action='store_true',help='clips data to avoid matplotlib warnings')
parser.add_argument('--seq',action='store_true',help='Run model predictions on batch steps')
parser.add_argument('--iter',action='store_true',help='Run model predictions on own output')

args = parser.parse_args()

batch, x_amt,y_amt, time_scale = ml_overfit.load_and_norm_data()

num_c = ml_overfit.get_num_channels()

print ("Creating model")
model = ml_overfit.create_model(x_amt,y_amt, num_c)
#lkfjds
print ('Loading weights')
model.load_weights('weights')


#Sequential prediction - predict each from previous steps labels
if args.seq or not args.iter:
    predictions = model.predict(batch)

#Iterative calculation - predict each from previous step of prediction
else:
    start_point = 0
    num_pred = batch.shape[0]

    predictions = np.empty(shape=(batch.shape))
    predictions[:start_point]=batch[:start_point]

    pred = batch[start_point][None]

    for i in range(start_point,num_pred):
        pred = model.predict(pred)
        predictions[i]=pred[0]

predictions = np.roll(predictions,-1,axis=0)


fig, plots = plt.subplots(2,3,sharex=True,sharey=True)

restart=False

def close_event(var):
    exit()
def keypress(event):
    global restart
    if event.key=='backspace':
        restart=True
    else:
        print (event.key)
fig.canvas.mpl_connect('close_event', close_event)
fig.canvas.mpl_connect('key_press_event', keypress)

#speedup looping
# predictions = np.swapaxes(predictions,3,1)
# batch = np.swapaxes(batch,3,1)

temp_scaling=.1
vel_scaling = 5e3
fill_val = .5

temp_p = predictions[:,:,:,0] * temp_scaling + .5
temp_a = batch[:,:,:,0] * temp_scaling + .5

predictions[:,:,:,1:] = predictions[:,:,:,1:] * vel_scaling + .5
batch[:,:,:,1:]  = batch[:,:,:,1:] * vel_scaling + .5

#Solely to avoid annoying matplotlib warnings
if args.clip:
    temp_p=np.clip(temp_p,0,1)
    temp_a=np.clip(temp_a,0,1)

    predictions = np.clip(predictions,0,1)
    batch = np.clip(batch,0,1)

vel_p  = np.pad(predictions[:,:,:,1:3],((0,0),(0,0),(0,0),(0,1)),constant_values=fill_val)
accel_p= np.pad(predictions[:,:,:,3:5],((0,0),(0,0),(0,0),(0,1)),constant_values=fill_val)

vel_a  = np.pad(batch[:,:,:,1:3],((0,0),(0,0),(0,0),(0,1)),constant_values=fill_val)
accel_a= np.pad(batch[:,:,:,3:5],((0,0),(0,0),(0,0),(0,1)),constant_values=fill_val)



#while loop allows restarts
t=0
while t < batch.shape[0]:
    if t==ml_overfit.TRAIN_REGION:
        print ('Purely predicted')
    elif t==130:
        print ('Unseen')
    print ('t=',t)
    frame_time = time.time()

    plots[0,0].imshow(temp_p[t])
    plots[0,1].imshow(vel_p[t])
    plots[0,2].imshow(accel_p[t])

    plots[1,0].imshow(temp_a[t])
    plots[1,1].imshow(vel_a[t])
    plots[1,2].imshow(accel_a[t])

    total_time = time.time() - frame_time
    sleep_time = .1 - total_time
    if sleep_time < .001:
        sleep_time = .001
    plt.pause(sleep_time)
    if restart:
        t=0
        restart=False
    else:
        t+=1
