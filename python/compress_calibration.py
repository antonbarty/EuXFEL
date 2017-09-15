#!/usr/bin/env python

import h5py
import sys

if len(sys.argv) < 2:
    print('Usage: compress_calibration.py <compressed.h5> [calibration_input.h5]')
    sys.exit()

calibration_file ='/gpfs/exfel/data/scratch/example_data/agipd_test_store/AGIPD/21082017T000000.h5'

# Write down the cells that have data
good_cells = [4,8,12,16,20,24,28]

f = h5py.File(calibration_file)
out_f = h5py.File(sys.argv[1])
out_f['input_calibration'] = calibration_file
out_f['good_cells'] = good_cells
out_f.create_dataset('offsets', (16,128,512,len(good_cells)),dtype='uint16')

module = 0
for i in range(1,5):
    for j in range(1,5):
        dset = '/Q%dM%d/Offset/0/data' % (i,j)
        for k in range(len(good_cells)):
            out_f['offsets'][module,:,:,k] = f[dset][:,:,good_cells[k],0]
        module+=1

out_f.close()


