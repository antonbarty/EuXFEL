#!/usr/bin/env python

import sys
import os
import h5py
import numpy as np
import glob
import multiprocessing as mp
import ctypes
import combine_modules

if len(sys.argv) < 2:
    print('Format: %s <run_number>'%sys.argv[0])
    sys.exit(1)
raw_path = '/gpfs/exfel/exp/SPB/201701/p002012/raw/r%.4d' % int(sys.argv[1])
with h5py.File('data/r0063_calib_small.h5', 'r') as f:
    offsets = f['offsets'][:].transpose(3,0,2,1)
good_cells = list(range(2,30,2))
hist_min = -100
hist_max = 300
ix, iy = np.indices((512,128))
hist_shape = (16,len(good_cells),512,128,hist_max-hist_min)

c = combine_modules.AGIPD_Combiner(raw_path, good_cells=good_cells)

def hist_worker(mod, parity, hist):
    dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/data'%mod
    np_hist = np.frombuffer(hist.get_obj(), dtype='u2').reshape(hist_shape)[mod]
    
    # For each file with module 'mod'
    for j, fname in enumerate(c.flist[mod]):
        with h5py.File(fname , 'r') as f:
            num_trains = f[dset_name].shape[0] // 60
            # For each train
            for t in range(num_trains):
                # For each good cell of the given parity
                for k,cell in enumerate(good_cells[parity::2]):
                    data = f[dset_name][t*60+cell,0,:,:].astype('i4') - offsets[k,mod]
                    data[data<hist_min] = hist_min
                    data[data>=hist_max] = hist_max-1
                    data -= hist_min
                    np_hist[k*2+parity, ix, iy, data] += 1
                
                if mod == 0:
                    sys.stderr.write('\rModule 0: (%d, %d)'%(j,t))

# 10.9 GB array for 14 cells and 400 bins
hist = mp.Array(ctypes.c_ushort, np.prod(hist_shape))
jobs = []
for i in range(32):
    p = mp.Process(target=hist_worker, args=(i//2, i%2, hist))
    jobs.append(p)
    p.start()
for j in jobs:
    j.join()
sys.stderr.write('\n')
hist = np.frombuffer(hist.get_obj(), dtype='u2').reshape(hist_shape)

if not os.path.isdir('data/'):
    os.mkdir('data')
with h5py.File('data/histograms_r%.4d.h5'%int(sys.argv[1]), 'w') as f:
    f['histograms'] = hist


