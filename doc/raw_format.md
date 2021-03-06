# Description of the EuXFEL Raw data format

The raw data is written in multiple files separated by detector and sliced up
in 250 trains per file at the moment. For example `RAW-R0040-AGIPD04-S00004.h5`
corresponds to run 40, detector module 4 and slice 4.

The detector data inside the hdf5 files is in a group with a name that depends
on the detector module. For example `RAW-R0040-AGIPD04-S00004.h5` will have the
data in the group `/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/4CH0:xtdf/image/data` while
`RAW-R0040-AGIPD10-S00004.h5` will have the data in
`/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/10CH0:xtdf/image/data`. In general is has the
format `'/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/data' % (module)`.

The detector data has the shape `(250*60, 2, 512, 128)`, corresponding to 250
trains with 60 elements each. All the data in `(:,1,:,:)` is pure noise.
The data in `(::2,0,:,:)` corresponds to the raw ADU values of the detector. This
is also known as the 'analog' signal by the detector group.

The data in `(1::2,0,:,:)` corresponds to the gain mode values for each
pixel. This is also known as the 'digital' signal by the detector group.
This is *not* a simple `[0,1,2]` value but it has to be thresholded to
transform it into that.

We'll call the first index the frame_id. The pulse train number can be
calculated at `(frame_id % 60)`. 

The detector has 352 cells but for this experiment we're only using the first 30
cells. You can find the cell number for a given frame_id with `cell = (frame_id %
60)/2`. The reason for the `/2` is due to the fact that the 'analog' and
'digital' are split over two frame_ids.

During this experiment the detector is running 4x faster than the accelerator.
Recently vetoeing is being used to overwrite every other pulse.

For "technical reasons" the first cell never contains X-ray data.

This means that without vetoing the cells that see X-rays are `[4,8,...,28]`, for
a total of 7 images per train. This correspond to frame_ids `[8,16,...,56]`.

With vetoing the cells that see X-rays are `[2,4,...,28]`, for a total of 14
images per train (as long as the trains have 14 or more X-ray pulses). 



