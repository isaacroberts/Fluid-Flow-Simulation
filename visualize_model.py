import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import ml_overfit

import time


x_amt = 128
y_amt = 128
num_c = 6

print ("Creating model")
model = ml_overfit.create_model(x_amt,y_amt, num_c)
#lkfjds
print ('Loading weights')
model.load_weights('weights')



print (weights)
