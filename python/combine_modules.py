#!/usr/bin/env python

import sys
import h5py
import numpy as np
import glob

class AGIPD_Combiner():
    '''
    Interface to get frames interactively
    Initially specify path to folder with raw data
    Then use get_frame(num) to get specific frame
    '''
    def __init__(self, folder_path, verbose=0, good_cells=[4,8,12,16,20,24,28]):
        self.verbose = verbose
        self.good_cells = np.array(good_cells)*2
        self._make_flist(folder_path)
        self._get_nframes_list()
        self.frame = np.empty((16,512,128))

    def _make_flist(self, folder_path):
        self.flist = np.array([np.sort(glob.glob('%s/RAW-*-AGIPD%.2d*.h5'%(folder_path, r))) for r in range(16)])
        try:
            assert len(self.flist.shape) == 2
        except AssertionError:
            sys.stderr.write('Each module does not have the same number of files\n')
            raise
        if self.verbose > 0:
            print('%d files per module\n' % len(self.flist[0]))

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
                        dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/trainId'%i
                    except KeyError:
                        print(fname)
                        raise
        try:
            assert np.all(module_nframes == module_nframes[0])
        except AssertionError:
            sys.stderr.write('Not all modules have the same frames')
            raise
        if self.verbose > -1:
            print('%d good frames in run' % module_nframes[0])
        self.nframes = module_nframes[0]
        self.nframes_list = np.cumsum(self.nframes_list)

    def _get_frame(self, num, type='frame'):
        if num > self.nframes or num < 0:
            print('Out of range')
            return
        
        cell_ind = num % len(self.good_cells)
        train_ind = num // 60
        
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
            with h5py.File(self.flist[i,file_num], 'r') as f:
                dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/data'%i
                self.frame[i] = f[dset_name][frame_num,0]
        return self.frame

    def get_frame(self, num):
        return self._get_frame(num,type='frame')

    def get_gain(self, num):
        return self._get_frame(num,type='gain')


