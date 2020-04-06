import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import os
import scipy
import scipy.spatial

folder = '../fluidity/fluidity-master/examples/lock_exchange/'

series = 'lock_exchange'

file_ct = 0
file_no=0

done=False
while not done:
    filename = folder+series+'_'+str(file_no)+".vtu"
    if os.path.exists(filename):
        file_no+=1
        #File count is length of file array, not index of last file
        file_ct = file_no
    else:
        done=True

print ("Filect = ",file_ct)

arrays = ['Pressure','Temperature','Velocity']
second_dim=[1, 1, 3]
set_second_dim={'Pressure':1,'Temperature':1,'Velocity':2}
mix = [True, True, True]
# arrays = ['Temperature']
# second_dim=[1]

dict = {}
time = np.zeros(shape=(file_ct))

def dist_to_line(a,b,c, use_magn=False):
    #From point A to the line between B and C
    #I assume third dim is (x,y)

    #distance between A and line BC :
    # | (By - Cy)Ax - (Bx-Cx)Ay + BxCy - ByCx | / (dist(B,C))
    #Ignore divide by dist since final distances will be divided
    # (B - C) * A.t + B * inv(C)

    # (B - C)
    dbc = b - c
    # # (B-C) * A.t
    dist = dbc[:,:,1] * a[:,:,0] - dbc[:,:,0] * a[:,:,1]
    # + B * inv(C)
    dist += b[:,:,0] * c[:,:,1] - b[:,:,1] * c[:,:,0]
    if use_magn:
        dist /= (((b-c)**2).sum(axis=2))**.5
    return dist


xmax = -1e6
xmin = 1e10
ymax = -1e6
ymin = 1e10
for file_no in range(file_ct):
    filename = folder+series+'_'+str(file_no)+".vtu"
    # print (filename)

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    for p in range(reader.GetOutput().GetNumberOfPoints()):
        x,y,_ = reader.GetOutput().GetPoint(p)
        if x > xmax:
            xmax = x
        if x < xmin:
            xmin = x
        if y > ymax:
            ymax = y
        if y < ymin:
            ymin = y


x_amt = 128
y_amt = 128

shape = (file_ct, x_amt,y_amt)

for i in range(len(arrays)):
    name = arrays[i]
    if second_dim[i]==1:
        dict[name]=np.empty(shape=(file_ct,x_amt,y_amt))
    else:
        dict[name]=np.empty(shape=(file_ct,x_amt,y_amt,second_dim[i]))

grid_points = np.mgrid[0:x_amt,0:y_amt]
grid_points = np.swapaxes(grid_points,0,2)
grid_points = grid_points.reshape(-1,2)
# (X*Y) * 2
print ('points')
print (grid_points)
print (grid_points.shape)

# downsample = 2

for file_no in range(file_ct):
    filename = folder+series+'_'+str(file_no)+".vtu"
    # print (filename)

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()


    print ('----------',file_no,'-----------')
    # print (reader.GetOutput())
    output = reader.GetOutput()

    subdict={}
    for a_name in arrays:
        arr= vtk_to_numpy(output.GetPointData().GetArray(a_name))
        subdict[a_name]=arr

    time[file_no]=vtk_to_numpy(output.GetPointData().GetArray('Time'))[0]

    point_num = output.GetNumberOfPoints()
    points = np.zeros(shape=(point_num,2))
    for p in range(point_num):
        x,y,_ = output.GetPoint(p)
        xIx = int((x-xmin)/(xmax-ymin) * (x_amt-1))
        yIx = int((y-ymin)/(ymax-ymin) * (y_amt-1))

        points[p,0]=xIx
        points[p,1]=yIx

    # print ('orig points')
    # print (points.shape)

    triangles = scipy.spatial.Delaunay(points)

    if any(mix):
        #Px3x2
        tripoints = triangles.points[triangles.simplices]
        # print (tripoints.shape)
        #Subtract each vertices of the triangle from each other vertices
        #Trivecs: Px3x2

        #Divide by magnitude of vector (linalg.norm is distance)
        # magn = np.linalg.norm(trivecs,axis=2)

        b = np.roll(tripoints,-1,axis=1)
        c = np.roll(tripoints,-2,axis=1)


        dists = dist_to_line(tripoints, b, c, use_magn = False)
        #P x 3


        simpl_points = triangles.find_simplex(grid_points)
        # print (simpl_points.max())
        # print (simpl_points.shape)

        #Distance between point and line is (Line dot Point + offset) over Magnitude(line)
        #Magnitude (line)= 1 since its unit vector

        map = dist_to_line(grid_points[:,None,:], b[simpl_points], c[simpl_points], use_magn = False)
        map = map / dists[simpl_points]
        map /= map.sum(axis=1)[:,None]

        na = ~np.isfinite(map)
        if na.any():
            print ('na_ct=',na.sum())
            map[na.any(axis=1)]=.333


    all_simplex = triangles.simplices[simpl_points,:]
    # print (all_simplex.shape)
    for i in range(len(arrays)):
        a = arrays[i]
        # print (a)
        # print (subdict[a].shape)
        means = subdict[a][all_simplex]
        if mix[i]:
            if second_dim[i]==1:
                means *= map
            else:
                means *= map[:,:,None]
            means = means.sum(axis=1)
        else:
            means = means.mean(axis=1)
        # print (means)
        # print (means.shape)
        if second_dim[i] == 1:
            means = means.reshape(x_amt,y_amt)
        else:
            means = means.reshape(x_amt,y_amt,second_dim[i])
        dict[a][file_no]=means
        # mean = subdict[a][]


output_folder = 'NpArrays/'
try:
    os.mkdir(output_folder)
except Exception as e:
    pass



for k,v in dict.items():
    second_dim = set_second_dim[k]
    if second_dim==1:
        v = v[:,:,:,None]
    else:
        v = v[:,:,:,:second_dim]
    print (k,v.shape)
    np.save(output_folder+k,v)
np.save(output_folder+'Time',time)
