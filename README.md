Ouster SDK webinar examples
===========================
Tom Slankard <tom.slankard@ouster.io>

This repository contains examples presented during Ouster's 2024-12-12 SDK webinar.

These examples require ouster-sdk 0.13.1.

To use these examples, create a virtual environment using a version of Python from 3.8 to 3.12 and install ouster-sdk.

Here are the instructions for Linux or macOS:

    $ git clone https://github.com/ouster-lidar/ouster-sdk-webinar.git
    $ cd ouster-sdk-webinar
    $ python3.12 -m venv venv                # or other python version
    $ source venv/bin/activate
    $ pip install ouster-sdk==0.13.1

Obtain sample data by navigating to
https://static.ouster.dev/sensor-docs/#sample-data and downloading an OSF file
from one of the drives linked on that page. Rename the file to `webinar.osf`
and place it in the same folder as the examples.

At this point, you should have the following in your directory:

    $ ls
    2024-12-12  LICENSE  README.md  webinar.osf

From there, it should be possible to run any of the examples.

    $ python 2024-12-12/00_open_source.py    # and so on
