from ouster.sdk import open_source

source_path = 'webinar.osf'

# open the source - it may be a sensor or file
src = open_source(source_path)

# iterate through the LidarScan objects in the source and print them
for scan in src:
    print(scan)
