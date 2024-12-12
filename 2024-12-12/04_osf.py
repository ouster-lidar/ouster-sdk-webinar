from ouster.sdk import open_source
import ouster.sdk.osf as osf
import numpy as np


source_path = 'webinar.osf'
source = open_source(source_path)


# create a Writer object
with osf.Writer('example.osf', source.metadata) as writer:

    # Use it to save LidarScan objects to the specified OSF file
    for scan in source:

        # Add a custom field while we're at it
        scan.add_field('customfield', dtype=np.float64)

        # Giving each scan row its own value interpolated from 0 to 1
        column = np.linspace(0, 1, source.metadata.h)
        scan.field('customfield')[:] = np.repeat(column, source.metadata.w).reshape(source.metadata.h, source.metadata.w)
        writer.save(0, scan)
        break


# Print out what we got
for scan in open_source('example.osf'):
    print(scan.field('customfield'))
