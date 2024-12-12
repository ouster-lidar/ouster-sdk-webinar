from ouster.sdk import viz, open_source
import numpy as np

source_path = 'example.osf'

# get the first scan from the source
source = open_source(source_path)
scan = next(iter(source))

# create a cloud from the LidarScan RANGE field
cloud = viz.Cloud(source.metadata)
cloud.set_range(scan.field("RANGE"))
cloud.set_point_size(3)

# set the cloud color to the custom field value
color = scan.field("customfield")
cloud.set_key(color)

# create the viewer and add the cloud to it
v = viz.PointViz("Hello World!")
viz.add_default_controls(v)
v.add(cloud); v.update(); v.run()
