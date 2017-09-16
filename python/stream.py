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
    
    def __init__(self, calib_fnam, rundir, gain_factors=[1., 35., 4.2]):
        """
        Calculate good frame indexs, find and align module filenames...
        """
        # good pulses in train
        self.good_pulses = np.arange(8,60,8)
        
        # good pulses in file
        self.good_frames = np.array([self.good_pulses + t for t in range(0, 15000, 60)]).ravel()
        
        f = h5py.File(calib_fnam)
        # transpose the offsets file for easier + faster access
        # (cell, gain, module, ss, fs)
        self.offset    = np.transpose(f['offset'][()],    (2, 1, 0, 3, 4)).astype(np.float32)
        # (cell, thresh, module, ss, fs)
        self.threshold = np.transpose(f['threshold'][()], (2, 1, 0, 3, 4)).astype(np.float32)

        # gain multiplication factors
        self.gain_factors = gain_factors
        
        # grab the file names for each of the modules
        self.mod_fnams = self.get_mods(rundir)
        
        # initialise the iterator stuff
        self.index     = 0
        self.s_index   = 0
        self.__len__   = len(self.good_frames) * len(self.mod_fnams[0])
          
        # create an empty array for holding the single frames
        self.shape  = (16, 512, 128)
        self.frame  = np.empty(self.shape, dtype=np.float32)
        self.gain   = np.empty(self.shape, dtype=np.float32)
         
    def calibrate_frame(self, frame, gain, cell):
        """
        offset calib[mod, fs, ss, cell no.], cell no. = (i/2)%30
        """
        g0 = gain < self.threshold[cell, 0]
        g1 = (~g0) * (gain < self.threshold[cell, 1])
        g2 = (~g0) * (~g1)
        
        frame[g0] -= self.offset[cell, 0][g0]
        frame[g1] -= self.offset[cell, 1][g1]
        frame[g2] -= self.offset[cell, 1][g2]

        frame[g0] *= self.gain_factors[0]
        frame[g1] *= self.gain_factors[1]
        frame[g2] *= self.gain_factors[2]
        
        return frame
        #return g0.astype(np.float32)+2*g1.astype(np.float32)+3*g2.astype(np.float32)
    
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
        # file index
        self.s_index = i // self.__len__
        
        # inter file index
        ii           = self.good_frames[i % self.__len__]
        cell_ids = np.zeros((16), dtype=np.uint8)
        
        # get the modules for this frame
        for m in range(16):
            DATAPATH = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%iCH0:xtdf/image' % m
            f = h5py.File(self.mod_fnams[m][self.s_index], 'r')
            self.frame[m] = f[DATAPATH + '/data'][ii,   0].astype(np.float32)
            self.gain[m]  = f[DATAPATH + '/data'][ii+1, 0].astype(np.float32)
            cell_ids[m] = f[DATAPATH + '/cellId'][ii, 0]//2
            f.close()
        
        print(ii, cell_ids[0])
        
        # make sure the memory cell ids are all the same
        assert(np.all(cell_ids == cell_ids[0]))
        
        # calibrate the frame
        self.frame = self.calibrate_frame(self.frame, self.gain, cell_ids[0])
        
        return self.frame.copy()

        
if __name__ == '__main__':
    #calib  = h5py.File('../calibration/small_calibration.h5', 'r')['offsets'][()]
    #calib  = h5py.File('/gpfs/exfel/exp/SPB/201701/p002012/scratch/filipe/offset_and_threshold.h5', 'r')['offset'][()]
    cfnam  = '/gpfs/exfel/exp/SPB/201701/p002012/scratch/filipe/offset_and_threshold.h5'
    rundir = '/gpfs/exfel/exp/SPB/201701/p002012/raw/r0005'
    gfnam  = '/gpfs/exfel/u/scratch/SPB/201701/p002012/amorgan/EuXFEL/agipd_hmg2_oy0_man.geom'
    
    # example
    stream  = Stream(cfnam, rundir)

    # assemble 100 frames (kind of slow) (event, module, ss, fs)
    frames  = stream[:100]

    # apply the geometry to each frame for viewing
    gframes = np.array([geom.apply_geom(gfnam, frame) for frame in frames])
    
