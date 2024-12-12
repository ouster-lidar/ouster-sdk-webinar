import copy
from ouster.sdk import open_source
from ouster.sdk.client import destagger
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


# read a LidarScan from the source
source_path = 'webinar.osf'
source = open_source(source_path)
scan = next(iter(source))

# retrieve the RANGE and NEAR_IR fields from the scan
range_field = scan.field("RANGE")
near_ir_field = scan.field("NEAR_IR")

# destagger the fields
rng = destagger(source.metadata, range_field)
near_ir = destagger(source.metadata, near_ir_field)

# set up a plot of the images along with histograms of their values
plt.subplot(2, 2, 1)
viridis = copy.copy(matplotlib.colormaps['viridis'])
viridis.set_bad((0, 0, 0))
im = plt.imshow(rng, norm=colors.LogNorm(), cmap=viridis, aspect='auto')
plt.colorbar(im, orientation='horizontal')

plt.subplot(2, 2, 2)
plt.hist(rng.flatten(), bins=256, range=[1, np.percentile(rng, 98)])

plt.subplot(2, 2, 3)
im = plt.imshow(near_ir, norm=colors.LogNorm(), cmap='gray', aspect='auto')
plt.colorbar(im, orientation='horizontal')

plt.subplot(2, 2, 4)
plt.hist(near_ir.flatten(), bins=256, range=[1, np.percentile(near_ir, 98)])

# show the plot
plt.margins(0.05, tight=True)
plt.show()
