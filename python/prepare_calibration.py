#!/usr/bin/env python

import h5py
import sys
import glob
import numpy as np

if len(sys.argv) < 2:
    print('Takes a directory with 32 files which match *AGIPD??*.h5')
    print('produces a single calibration output file')
    print('')
    print('Usage: prepare_calibration.py <calib_out.h5> [calib_dir]')
    sys.exit()

calibration_dir = '/gpfs/exfel/exp/SPB/201701/p002012/scratch/user/kuhnm/dark'
if len(sys.argv) > 2:
    calibration_dir = sys.argv[2]

offset_files = sorted(glob.glob(calibration_dir+'/*dark*AGIPD??_agipd*.h5'))
#gain_files = sorted(glob.glob(calibration_dir+'/*combin*AGIPD??*.h5'))
if(len(offset_files) != 16):
    print('Error: found %d offset files instead of 16.' % (len(offset_files)))
    sys.exit()

# if(len(gain_files) != 16):
#     print('Error: found %d gain files instead of 16.' % (len(gain_files)))
#     sys.exit()

out_f = h5py.File(sys.argv[1])
out_f.create_dataset('offset', (16,3,30,512,128),dtype='uint16')
out_f.create_dataset('threshold', (16,2,30,512,128),dtype='uint16')
#out_f.create_dataset('gain', (16,3,30,512,128),dtype='float16')

for i in range(16):
    f = h5py.File(offset_files[i])
    if i < 8:
        # For low module number we need to invert the 512 pixels axis
        out_f['offset'][i,:,:,:,:] = np.tranpose(np.array(f['offset'])[:,::-1],(1,2,4,3))
        out_f['threshold'][i,:,:,:,:] = np.tranpose(np.array(f['threshold'])[:,::-1],(1,2,4,3))
    else:
        # For high module number we need to invert the 128 pixels axis
        out_f['offset'][i,:,:,:,:] = np.tranpose(np.array(f['offset'])[::-1,:],(1,2,4,3))
        out_f['threshold'][i,:,:,:,:] = np.tranpose(np.array(f['threshold'])[::-1,:],(1,2,4,3))
    f.close()
#    f = h5py.File(gain_files[i])
#    out_f['gain'][i,:,:,:,:] = np.array(f['gain'])
#    f.close()

    # out_f['offsets'][i,:,:,:,:] = np.tranpose(np.array(in_files[i]['offset']),(0,1,3,2))   
    # out_f['thresholds'][i,:,:,:,:] = np.tranpose(np.array(in_files[i]['thewsholds']),(0,1,3,2))
    # out_f['gains'][i,:,:,:,:] = np.tranpose(np.array(in_files[i]['thewsholds']),(0,1,3,2))
    
out_f.close()

