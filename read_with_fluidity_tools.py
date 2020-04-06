import vtktools
import numpy as np
import matplotlib.pyplot as plt
import os

folder = '../fluidity/fluidity-master/examples/lock_exchange/'

series = 'lock_exchange'

file_ct = 0
n=0

done=False
while not done:
    filename = folder+series+'_'+str(n)+".vtu"
    if os.path.exists(filename):
        n+=1
        #File count is length of file array, not index of last file
        file_ct = n
    else:
        done=True

print ("Filect = ",file_ct)

arrays = ['Pressure','Temperature','TemperatureInterpolationErrorBound',
        'Time','Velocity','VelocityInterpolationErrorBound','GravityDirection','Viscosity']
dict = {}

for n in range(file_ct):
    filename = folder+series+'_'+str(n)+".vtu"
    # print (filename)

    ug = vtktools.vtu(filename)

    print ('----------',n,'-----------')
    # print (reader.GetOutput())

    print (ug.GetFieldNames())

    p = ug.GetScalarField('Pressure')
    # print (p)
    print (p.shape[0])
    v = ug.GetVectorField('Velocity')
    # print (v)
    # print (v.shape)

        # if n==0:
        #     dict[a_name] = arr
        # else:
        #     dict[a_name] = np.concatenate((dict[a_name],arr),axis=0)



print ('------------dict ------------')
# print (dict)
for k,v in dict.items():
    print (k,':',v.shape)

for a in arrays:
    print (a)
    plt.plot(dict[a])
    plt.show()
