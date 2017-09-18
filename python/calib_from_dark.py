#!/usr/bin/env python

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print('Format: %s <run_number>' % sys.argv[0])
    sys.exit(1)

run = int(sys.argv[1])
good_cells = list(range(2,30,2))

out_file = 'r%.4d_calib_small.h5'%run
fout = h5py.File(out_file,'w')
fout.create_dataset('offsets', (16,128,512,len(good_cells)),dtype='uint16')

for m in range(16):    
    dark_sum = np.zeros((len(good_cells),512,128))
    dark_file = '/gpfs/exfel/exp/SPB/201701/p002012/raw/r%.4d/RAW-R%.4d-AGIPD%02d-S00001.h5' % (run,run,m)
    f = h5py.File(dark_file)
    dset = f['/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/data' % (m)]    
    for i in range(dset.shape[0]//60):
        for j in range(len(good_cells)):
            index = 60*i+good_cells[j]*2
            dark_sum[j,:,:] += dset[index,0,:,:]
    sys.stderr.write('\r(%.4d, %.4d)' % (i, m))

    dark_sum /= dset.shape[0]/60
    fout['offsets'][m,:,:,:] = np.transpose(dark_sum,(2,1,0))
sys.stderr.write('\n')


