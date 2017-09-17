#!/usr/bin/env python
"""
Designed to be run with 4 cores
everyone computes their own histograms 
for each quadrant then we collect them
at the end. I do not know if 2 cores 
trying to access the same event is slow...
"""

import sys
import os
import argparse
import numpy as np

hist_dtype   = np.uint16
buffer_dtype = np.float32
cspad_shape  = (16, 512, 128)


cfnam  = '/gpfs/exfel/exp/SPB/201701/p002012/scratch/filipe/offset_and_threshold.h5'
run    = 48


if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    from stream import Stream
    stream  = Stream(cfnam, run)

    print(stream.__len__)
    
    #-----------------------------
    # Actual meat
    #-----------------------------
    #frames = stream[rank:stream.__len__:size]
    try :
        frames = stream[rank :stream.__len__:size]
        frames[frames<50] = 0.
     
        powder  = np.sum(frames, axis=0)
    except Exception as e :
        print(e)
        powder = np.zeros((16, 512, 128), dtype=np.float32)
    
    comm.Reduce(powder.copy(), powder, root = 0)

    if rank == 0 :
        import h5py
        f = h5py.File('powder_'+str(run)+'.h5')
        if 'powder' in f :
            del f['powder']

        f['powder'] = powder
        f.close()
