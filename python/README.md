# stream.py
Basic usage:

```bash
ssh -X max-cfel
cd /gpfs/exfel/exp/SPB/201701/p002012/scratch/amorgan/EuXFEL/python
source  /gpfs/cfel/cxi/common/public/cfelsoft-rh7-public/setup.sh
module load cfel-anaconda/py3-4.4.0
ipython
%run stream
%gui qt
import pyqtgraph as pg
pg.show(gframes)
```

