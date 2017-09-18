#!/usr/bin/env python

import sys
import h5py
import numpy as np
import glob
import multiprocessing as mp
import ctypes
import geom

class AGIPD_Combiner():
    '''
    Interface to get frames interactively
    Initially specify path to folder with raw data
    Then use get_frame(num) to get specific frame
    '''
    def __init__(self, folder_path, verbose=0, good_cells=[4,8,12,16,20,24,28], geom_fname=None,
                 calib_file='/gpfs/exfel/exp/SPB/201701/p002012/scratch/filipe/offset_and_threshold.h5'):
        self.verbose = verbose
        self.good_cells = np.array(good_cells)*2
        self.geom_fname = geom_fname
        if self.geom_fname is not None:
            self.x, self.y = geom.pixel_maps_from_geometry_file(geom_fname)
        self._make_flist(folder_path)
        self._get_nframes_list()
        self.frame = np.empty((16,512,128))
        self.powder = None
        self.calib = h5py.File(calib_file,'r')
        
    def _make_flist(self, folder_path):
        self.flist = np.array([np.sort(glob.glob('%s/RAW-*-AGIPD%.2d*.h5'%(folder_path, r))) for r in range(16)])
        try:
            assert len(self.flist.shape) == 2
        except AssertionError:
            print('Each module does not have the same number of files')
            print([len(f) for f in self.flist])
        if self.verbose > 0:
            print('%d files per module' % len(self.flist[0]))

    def _get_nframes_list(self):
        module_nframes = np.zeros((16,), dtype='i4')
        self.nframes_list = []
        for i in range(16):
            for fname in self.flist[i]:
                with h5py.File(fname, 'r') as f:
                    try:
                        dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/data'%i
                        module_nframes[i] += f[dset_name].shape[0] / 60 * len(self.good_cells)
                        if i == 0:
                            self.nframes_list.append(f[dset_name].shape[0])
                        #dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/trainId'%i
                    except KeyError:
                        print(fname)
                        raise
        try:
            assert np.all(module_nframes == module_nframes[0])
        except AssertionError:
            print('Not all modules have the same frames')
        if self.verbose > -1:
            print('%d good frames in run' % module_nframes[0])
        self.nframes = module_nframes[0]
        self.nframes_list = np.cumsum(self.nframes_list)

    def _calibrate(self, data, gain, module, cell):        
        data = np.float32(data)
        high_gain = gain < self.calib['threshold'][module,0,cell,:,:]
        low_gain = gain > self.calib['threshold'][module,1,cell,:,:]
        medium_gain =  ~high_gain * ~low_gain
        data -= self.calib['offset'][module,0,cell,:,:] * high_gain
        data -= self.calib['offset'][module,1,cell,:,:] * medium_gain
        data -= self.calib['offset'][module,2,cell,:,:] * low_gain
        #data[medium_gain] *= 45
        #data[low_gain] *= 45 * 3.8
        data[data < -100] = 0
        #data[data > 10000] = 10000
        return data

    def _threshold(self, gain, module, cell):        
        high_gain = gain < self.calib['threshold'][module,0,cell,:,:]
        low_gain = gain > self.calib['threshold'][module,1,cell,:,:]
        medium_gain =  ~high_gain * ~low_gain
        return low_gain*2+medium_gain*1
        
    def _get_frame(self, num, type='frame', calibrate=False, threshold=False, sync=True):
        if num > self.nframes or num < 0:
            print('Out of range')
            return
        
        if not sync:
            shift = 0
        cell_ind = num % len(self.good_cells)
        train_ind = num // len(self.good_cells)
        
        if type == 'frame':
            ind = self.good_cells[cell_ind] + train_ind * 60
        elif type == 'gain':
            ind = self.good_cells[cell_ind] + train_ind * 60 + 1
        else:
            raise ValueError
        
        file_num = np.where(ind < self.nframes_list)[0][0]
        if file_num == 0:
            frame_num = ind 
        else:
            frame_num = ind - self.nframes_list[file_num-1]
        for i in range(16):
            if len(self.flist[i]) == 0:
                self.frame[i] = np.zeros_like(self.frame[0])
                continue
            with h5py.File(self.flist[i][file_num], 'r') as f:
                dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/data'%i
                cell_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/cellId'%i
                train_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/trainId'%i
                if sync:
                    if i == 0:
                        trainid = f[train_name][frame_num].astype('i8')[0]
                        shift = 0
                    else:
                        shift = (trainid - f[train_name][frame_num].astype('i8')[0]) * 60
                data = f[dset_name][frame_num+shift,0]
                if calibrate:
                    data = self._calibrate(data,
                                           f[dset_name][frame_num+shift+1,0],
                                           i, self.good_cells[cell_ind]//2)
                if threshold:
                    data = self._threshold(data, i, cell_ind)
                self.frame[i] = data
        if self.geom_fname is None:
            return self.frame
        else:
            return geom.apply_geom_ij_yx((self.x, self.y), self.frame)

    def get_frame(self, num, calibrate=False, sync=True):
        return self._get_frame(num,type='frame', calibrate=calibrate, sync=sync)

    def get_gain(self, num, threshold=False, sync=True):
        return self._get_frame(num,type='gain', calibrate=False, threshold=threshold, sync=sync)

    def get_powder(self):
        if self.powder is not None:
            print('Powder sum already calculated')
            return self.powder
        
        powder_shape = (len(self.good_cells),) + self.frame.shape
        powder = mp.Array(ctypes.c_double, len(self.good_cells)*self.frame.size)
        jobs = []
        for i in range(16):
            p = mp.Process(target=self._powder_worker, args=(i, powder, powder_shape))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        sys.stderr.write('\n')
        self.powder = np.frombuffer(powder.get_obj()).reshape(powder_shape)
        
        return self.powder

    def _powder_worker(self, i, powder, shape):
        dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/data'%i
        np_powder = np.frombuffer(powder.get_obj()).reshape(shape)
        
        # For each file with module i
        for j in range(len(self.flist[i])):
            with h5py.File(self.flist[i][j] , 'r') as f:
                # For each cell
                for k,cell in enumerate(self.good_cells):
                    ind = np.zeros((f[dset_name].shape[0],), dtype=np.bool)
                    ind[cell::60] = True
                    np_powder[k,i] += f[dset_name][cell::60,0,:,:].mean(0)
                    if i == 0:
                        sys.stderr.write('\rModule 0: (%d, %d)'%(j,k))
        for k in range(len(self.good_cells)):
            np_powder[k,i] /= len(self.flist[i])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Format: %s <run_number>'%sys.argv[0])
        sys.exit(1)
    path = '/gpfs/exfel/exp/SPB/201701/p002012/raw/r%.4d' % int(sys.argv[1])
    print('Calculating powder sum from', path)
    
    c = AGIPD_Combiner(path, good_cells=list(range(2,30,2)))
    c.get_powder()
    
    import os
    if not os.path.isdir('data'):
        os.mkdir('data')
    f = h5py.File('data/raw_powder_r%.4d.h5'%int(sys.argv[1]), 'w')
    f['powder'] = c.powder
    f.close()

