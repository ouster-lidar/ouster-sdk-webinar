import copy
from ouster.sdk import open_source
from ouster.sdk.client import destagger
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

source_path = 'webinar.osf'
source = open_source(source_path)
scan = next(iter(source))

# retrieve the RANGE fields from the scan
range_field = scan.field("RANGE")

# destagger the field
rng = destagger(source.metadata, range_field)

viridis = copy.copy(matplotlib.colormaps['viridis'])
viridis.set_bad((0, 0, 0))

# display them
plt.subplot(2, 1, 2)
plt.imshow(rng, norm=colors.LogNorm(), cmap=viridis, aspect='auto')
plt.xlabel("Destaggered")
plt.subplot(2, 1, 1)
plt.imshow(scan.field("RANGE"), norm=colors.LogNorm(), cmap=viridis, aspect='auto')
plt.xlabel("Staggered")
plt.show()
