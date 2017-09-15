import h5py
import numpy as np
import glob
from enum import Enum
import geom

def slice_to_range(s):
    if s.start is None :
        start = 0
    else :
        start = s.start
    if s.step is None :
        step = 1
    else :
        step = s.step
    if s.stop is None :
        stop = len(self)
    else :
        stop = s.stop
    return range(start, stop, step)

class Stream():
    
    def __init__(self, calib, rundir):
        # good pulses in train
        self.good_pulses = np.arange(8,60,8)
        
        # good pulses in file
        self.good_frames = np.array([self.good_pulses + t for t in range(0, 15000, 60)]).ravel()
        
        # transpose the calib file for easier + faster access
        self.calib = np.transpose(calib, (3, 0, 2, 1)).astype(np.float)
        
        # grab the file names for each of the modules
        self.mod_fnams = self.get_mods(rundir)
        
        # initialise the iterator stuff
        self.index     = 0
        self.s_index   = 0
        self.__len__   = len(self.good_frames) * len(self.mod_fnams[0])
          
        # create an empty array for holding the single frames
        self.shape  = (16, 512, 128)
        self.frame  = np.empty(self.shape, dtype=np.uint16)
         
    def calibrate_frame(self, frame, index):
        """
        offset calib[mod, fs, ss, cell no.], cell no. = (i/2)%30
        """
        return frame.astype(np.float) - self.calib[index%7]
    
    # make self indexable: self[0] or self[40:100:2] etc
    def __getitem__(self, args):
        if type(args) == int :
            return self.get_frame(args)
        elif type(args) == slice :
            frames = slice_to_range(args)
            return np.array([self.get_frame(i) for i in frames])
        else :
            print('what?')
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self.s_index = self.index 
        if self.index == self.__len__ :
            raise StopIteration
        self.index += 1
        return self.get_frame(self.index-1)
        
    def get_mods(self, rundir):
        mods = []
        for m in range(16):
            mod = glob.glob(rundir + '/*D'+str(m).zfill(2)+'*.h5') 
            mod.sort()
            mods.append(mod)
        return mods
    
    def get_frame(self, i):
        self.s_index = i // self.__len__
        ii           = self.good_frames[i % self.__len__]
        print(ii)
        for m in range(16):
            DATAPATH = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%iCH0:xtdf/image/data' % m
            self.frame[m] = h5py.File(self.mod_fnams[m][self.s_index], 'r')[DATAPATH][ii, 0]
        self.frame = self.calibrate_frame(self.frame, ii)
        return self.frame.copy()

        
if __name__ == '__main__':
    #calib  = h5py.File('../calibration/small_calibration.h5', 'r')['offsets'][()]
    calib  = h5py.File('/gpfs/exfel/exp/SPB/201701/p002012/scratch/filipe/r0008_calib_small.h5', 'r')['offsets'][()]
    rundir = '/gpfs/exfel/exp/SPB/201701/p002012/raw/r0005'
    gfnam  = '/gpfs/exfel/u/scratch/SPB/201701/p002012/amorgan/EuXFEL/agipd_hmg2_oy0_man.geom'
    
    # example
    stream  = Stream(calib, rundir)
    frames  = stream[:100]
    gframes = np.array([geom.apply_geom(gfnam, frame) for frame in frames])
    
