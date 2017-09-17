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
hist_bins    = [-100, 200]


cfnam  = '/gpfs/exfel/exp/SPB/201701/p002012/scratch/filipe/offset_and_threshold.h5'
run    = 5


def hist_last_axis(data, R):
    """
    credit: https://stackoverflow.com/a/44155607
    """
    N = data.shape[-1]
    bins   = np.arange(R[0], R[1]+1, 1)
    n_bins = len(bins)-1
    data2D = data.reshape(-1, N)
    idx = np.searchsorted(bins, data2D, 'right') - 1
    
    bad_mask = (idx==-1) | (idx==n_bins)

    scaled_idx = n_bins * np.arange(data2D.shape[0])[:, None] + idx

    limit = n_bins * data2D.shape[0]
    scaled_idx[bad_mask] = limit

    counts = np.bincount(scaled_idx.ravel(), minlength=limit+1)[:-1]
    counts.shape = data.shape[:-1] + (n_bins,)
    return counts

if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    from stream import Stream
    stream  = Stream(cfnam, run)
    
    #-----------------------------
    # Actual meat
    #-----------------------------
    #frames = stream[rank:stream.__len__:size]
    hist = None
    for i in range(1000): 
        if (rank + 1000*i) >= stream.__len__ :
            break 
        
        try :
            frames = stream[rank + 1000*i :min(1000*(i+1), stream.__len__):size]
         
            if hist is None :
                hist  = hist_last_axis(np.transpose(frames, (1, 2, 3, 0)), hist_bins)
            else :
                hist += hist_last_axis(np.transpose(frames, (1, 2, 3, 0)), hist_bins)
        except Exception as e :
            print(e)
            break
    
    comm.Reduce(hist.copy(), hist, root = 0)

    if rank == 0 :
        import h5py
        f = h5py.File('hist.h5')
        if 'hist' in f :
            del f['hist']

        f['hist'] = hist
        f.close()
