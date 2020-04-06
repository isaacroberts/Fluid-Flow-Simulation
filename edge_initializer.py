
import numpy as np
import tensorflow as tf

from tensorflow.keras.initializers import Initializer

class EdgeInitializer(Initializer):
    def __init__(self,
        sep_channels= True,
        extra_dims=None,
        scale=.1):
        """
            Extra dims is what to do with the remaining kernels after 4
            Extra_dims can be a initializer of base Initializer, in which case kernel of 4 will be concatted to extra_dims output
            Or extra dims can be one of the following strings:
                ['repeat']
        """
        if extra_dims in ['repeat']:
            self.extra_dims_is_func = False
        elif isinstance(extra_dims,Initializer):
            self.extra_dims_is_func = True
        elif extra_dims is None:
            extra_dims = tf.keras.initializers.glorot_normal()
            self.extra_dims_is_func = True
        else:
            if isinstance(extra_dims,str):
                raise ValueError("Unrecognized extra_dims type:"+extra_dims)
            else:
                self.extra_dims_is_func = True

        self.extra_dims = extra_dims
        self.sep_channels = sep_channels
        self.scale = scale

    def __call__(self,shape,dtype):
        arr = self.edge_initializer(shape,dtype)
        # print ('Edge initializer')
        # print ('x0y0c0')
        # print (arr[0,0,0])
        # print ('edges')
        # print (arr[:,:,0,:4])
        # print ('full')
        # print (arr)
        # print (arr.shape)
        return arr

    def size(self,shape):
        if self.sep_channels:
            return self.shape[2] * 4
        else:
            return 4

    def edge_initializer(self,shape,dtype):
        """
        Creates pre-initialized edges
         -1 +1   +1 +1   +1 -1  -1 -1
         -1 +1,  -1 -1,  +1 -1, +1 +1
               + 4 random others
        To speed up initial training - you don't need 100 iterations to tell you where edges are


        Assumes a square kernel (shape[0]=shape[1])
        Allows odd kernel sizes, puts 0 down the middle of odd kernels
        Does not do gradients on bigger kernels
        """
        assert shape[0]==shape[1]

        if dtype is None:
            npdtype = np.float32
        else:
            npdtype = dtype.as_numpy_dtype()

        # arr = np.empty((shape[0],shape[0],shape[2],rotate_dim),npdtype)

        if self.extra_dims=='repeat':
            return self._finish_kernel(shape,npdtype)
        elif self.extra_dims_is_func:
            arr = self._base_kernel(shape,npdtype)

            newshape = (shape[0],shape[1],shape[2],shape[3]-arr.shape[3])
            other_kernels = self.extra_dims(newshape,dtype)
            return tf.concat([arr,other_kernels],axis=3)
        else:
            raise ValueError("Unrecognized extra dims type")

    def _base_kernel(self,shape,dtype,start=0):
        if self.sep_channels:
            amt = min(4 * shape[2], shape[3]-start)
            arr = np.zeros((shape[0],shape[0],shape[2],amt),dtype)
            return self._rotating_kernel(arr,0,amt=amt,channel=0)
        else:
            amt = min(4, shape[3]-start)
            arr = np.zeros((shape[0],shape[0],shape[2],amt),dtype)
            return self._rotating_kernel(arr,0,amt=amt,channel=None)

    def _finish_kernel(self,shape,dtype,start=0):
        if start > 0 :
            shape = list(shape)
            shape[3] -= start
            shape = tuple(shape)
        arr = np.empty(shape,dtype)
        if self.sep_channels:
            return self._rotating_kernel(arr,0,amt=None,channel=0)
        else:
            return self._rotating_kernel(arr,0,amt=None,channel=None)

    def _rotating_kernel(self,arr,start=0,amt=4,channel=None):
        """
            Sets any amount of first elements of kernel to rotating pattern
        """

        if channel == None:
            c = np.s_[:]
        else:
            c = channel

        xlen = arr.shape[0]
        l = int(xlen/2)

        if amt is None or start+amt > arr.shape[3]:
            amt = arr.shape[3] - start

        while amt > 0:
            #Set first half to +1
            arr[l:,:,c,0] =  self.scale
            arr[:-l,:,c,0]= -self.scale
            #If odd set middle to 0
            if amt%2==1:
                arr[l,:,c,0]=0
            #This is ugly but it prevents OOB on arbitrary number of channels
            if amt==1:
                return arr
            # Edge 1 = 90 degree rotation of Edge 0
            arr[:,:,c,1] = np.rot90(arr[:,:,c,0])
            if amt==2:
                return arr
            # Edge 2 = inverse of Edge 0
            arr[:,:,c,2] = -arr[:,:,c,0]
            if amt==3:
                return arr
            arr[:,:,c,3] = -arr[:,:,c,1]
            amt -= 4
            if self.sep_channels and channel is not None:
                c+=1
                if c > arr.shape[2]:
                    c=0
        return arr
