import numpy as np
import matplotlib.pyplot as plt
import time
import sys

folder = 'NpArrays/'


def close_event(var):
    exit(0)

# arrays = ['Pressure','Temperature','TemperatureInterpolationErrorBound',
#         'Time','Velocity','VelocityInterpolationErrorBound','GravityDirection','Viscosity']

arrays = ['Pressure','Temperature','Velocity','Time']

show_map = False
if show_map:
    try:
        map = np.load(folder+'map.npy')
        isna = ~np.isfinite(map).any(axis=3)
        map[isna]=0
    except:
        show_map=False

show_var_set = False
if len(sys.argv)>1:
    try:
        show_var = str(sys.argv[1])
        if show_var in arrays:
            show_var_set=True
        else:
            l = len(show_var)
            for a in arrays:
                if show_var == a[:l]:
                    show_var = a
                    show_var_set=True
                    break
    except:
        pass

if not show_var_set:
    show_var = 'Pressure'

print ('Showing',show_var)

arr = np.load(folder+show_var+'.npy')
print ('max arr',np.nanmax(arr))
print ('nanmean',np.nanmean(arr))
percentile = np.percentile(arr,95)
print ('percentile',percentile)
if percentile > 0:
    arr /= percentile

print (arr.shape)
if len(arr.shape) > 3:
    arr = np.linalg.norm(arr,axis=3)


fig = plt.figure(1)

fig.canvas.mpl_connect('close_event', close_event)

if show_map:
    sp = plt.subplot(121)
    sp2=plt.subplot(122)
else:
    sp = plt.subplot(111)

print (arr.mean())
plt.style.use('fast')
for t in range(arr.shape[0]):
    print (t,':')
    print (arr[t])
    sp.imshow(arr[t])
    if show_map:
        sp2=plt.imshow(map[t])
    # plt.show()
    plt.pause(.1)
