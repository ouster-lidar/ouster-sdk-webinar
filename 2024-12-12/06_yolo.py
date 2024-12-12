import argparse
from functools import partial

import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch

from ouster.sdk.client import ChanField, LidarScan, ScanSource, destagger, FieldClass, XYZLut
from ouster.sdk import open_source
from ouster.sdk.client._utils import AutoExposure, BeamUniformityCorrector
from ouster.sdk.viz import SimpleViz


class ScanIterator(ScanSource):

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    def __init__(self, scans: ScanSource, use_opencv=False):
        self._use_opencv = use_opencv
        self._metadata = scans.metadata

        # Since LidarScans are always fixed resolution imagery, we can create an efficient lookup table for
        # converting range data to XYZ point clouds
        self._xyzlut = XYZLut(self._metadata)

        # For nice viewing. Requires matplotlib
        self._generate_rgb_table()

        # Load yolo pretrained model.
        # The example runs yolo on both near infrared and reflectivity channels so we create two independent models
        self.model_yolo_nir = YOLO("yolov9c-seg.pt").to(device=self.DEVICE)
        #self.model_yolo_ref = YOLO("yolov9c-seg.pt").to(device=self.DEVICE)

        # Define classes to output results for.
        self.name_to_class = {}  # Make a reverse look up for convenience
        for key, value in self.model_yolo_nir.names.items():
            self.name_to_class[value] = key

        self.classes_to_detect = [
            self.name_to_class['person'],
            self.name_to_class['car'],
            self.name_to_class['truck'],
            self.name_to_class['bus']
        ]

        # Post-process the near_ir, and cal ref data to make it more camera-like using the
        # AutoExposure and BeamUniformityCorrector utility functions
        self.paired_list = [
            [ChanField.NEAR_IR, AutoExposure(), BeamUniformityCorrector(), self.model_yolo_nir],
            #[ChanField.REFLECTIVITY, AutoExposure(), BeamUniformityCorrector(), self.model_yolo_ref]
        ]

        # Map the self._update function on to the scans iterator
        # the iterator will now run the self._update command before emitting the modified scan
        self._scans = map(partial(self._update), scans)

    # Return the scans iterator when instantiating the class
    def __iter__(self):
        return self._scans

    def _generate_rgb_table(self):
        # This creates a lookup table for mapping the unsigned integer instance and class ids to floating point
        # RGB values in the range 0 to 1
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        # Make some colors for visualizing bounding boxes
        np.random.seed(0)
        N_COLORS = 256
        scalarMap = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1.0), cmap=mpl.pyplot.get_cmap('hsv'))
        self._mono_to_rgb_lut = np.clip(0.25 + 0.75 * scalarMap.to_rgba(np.random.random_sample((N_COLORS)))[:, :3], 0, 1)
        self._mono_to_rgb_lut = self._mono_to_rgb_lut.astype(np.float32)

    def mono_to_rgb(self, mono_img, background_img=None):
        """
        Takes instance or class integer images and creates a floating point RGB image with rainbow colors and an
        optional background image.
        """
        assert(np.issubdtype(mono_img.dtype, np.integer))
        rgb = self._mono_to_rgb_lut[mono_img % self._mono_to_rgb_lut.shape[0], :]
        if background_img is not None:
            if background_img.shape[-1] == 3:
                rgb[mono_img == 0, :] = background_img[mono_img == 0, :]
            else:
                rgb[mono_img == 0, :] = background_img[mono_img == 0, np.newaxis]
        else:
            rgb[mono_img == 0, :] = 0
        return rgb

    def _update(self, scan: LidarScan) -> LidarScan:
        stacked_result_rgb = np.empty((scan.h*len(self.paired_list), scan.w, 3), np.uint8)
        for i, (field, ae, buc, model) in enumerate(self.paired_list):

            # Destagger the data to get a human-interpretable, camera-like image
            img_mono = destagger(self._metadata, scan.field(field)).astype(np.float32)
            # Make the image more uniform and better exposed to make it similar to camera data YOLO is trained on
            ae(img_mono)
            buc(img_mono, update_state=True)

            # Convert to 3 channel uint8 for YOLO inference
            img_rgb = np.repeat(np.uint8(np.clip(np.rint(img_mono*255), 0, 255))[..., np.newaxis], 3, axis=-1)

            # Run inference with the tracker module enabled so that instance ID's persist across frames
            results: Results = next(
                model.track(
                    [img_rgb],
                    stream=True,  # Reduce memory requirements for streaming
                    persist=True,  # Maintain tracks across sequential frames
                    conf=0.1,
                    # Force the inference to use full resolution. Must be multiple of 32, which all Ouster lidarscans conveniently are.
                    # Note that yolo performs best when the input image has pixels with square aspect ratio. This is true
                    # when the OS0-128 is set to 512 horizontal resolution, the OS1-128 is 1024, and the OS2-128 is 2048
                    imgsz=[img_rgb.shape[0], img_rgb.shape[1]],
                    classes=self.classes_to_detect
                )
            ).cpu()

            # Plot results using the ultralytics results plotting. You can skip this if you'd rather use the
            # create_filled_masks functionality
            img_rgb_with_results = results.plot(boxes=True, masks=True, line_width=1, font_size=3)
            if self._use_opencv:
                # Save stacked RGB images for opencv viewing
                stacked_result_rgb[i * scan.h:(i + 1) * scan.h, ...] = img_rgb_with_results
            else:
                # Add a custom RGB results field to allow for displaying in SimpleViz
                scan.add_field(f"YOLO_RESULTS_{field}", destagger(self._metadata, img_rgb_with_results, inverse=True))

                # Alternative method for generating filled mask instance and class images
                # CAREFUL: These images are destaggered - human viewable. Whereas the raw field data in a LidarScan
                # is staggered.
                instance_id_img, class_id_img, instance_ids, class_ids = self.create_filled_masks(results, scan)

                # Example: Get xyz and range data slices that correspond to each instance id
                xyz_meters = self._xyzlut(scan.field(ChanField.RANGE))  # Get the xyz pointcloud for the entire LidarScan
                range_mm = scan.field(ChanField.RANGE)

                # It's more intuitive to work in human-viewable image-space so we choose to destagger the xyz and range data
                xyz_meters = destagger(scans._metadata, xyz_meters)
                range_mm = destagger(scans._metadata, range_mm)
                valid = range_mm != 0  # Ignore non-detected points
                for instance_id in instance_ids:
                    data_slice = (instance_id_img == instance_id) & valid
                    xyz_slice = xyz_meters[data_slice, :]  # The xyz data corresponding to an instance id
                    range_slice_mm = range_mm[data_slice]  # The range data corresponding to an instance id
                    # Example: Calculate the median range and xyz location to each detected object
                    print(f"ID {instance_id}: {np.median(range_slice_mm)/1000:0.2f} m, {np.array2string(np.median(xyz_slice, axis=0), precision=2)} m")

                # Add the data to the LidarScan for visualization. Always re-stagger (inverse=True)
                # the data to put it in the correct columns of the LidarScan. SimpleViz destaggers the data for human
                # viewing.
                scan.add_field(f"INSTANCE_ID_{field}", destagger(self._metadata, instance_id_img, inverse=True))
                scan.add_field(f"CLASS_ID_{field}", destagger(self._metadata, class_id_img, inverse=True))

                scan.add_field(f"RGB_INSTANCE_ID_{field}", destagger(self._metadata, self.mono_to_rgb(instance_id_img, img_mono), inverse=True))


        # Display in the loop with opencv
        if self._use_opencv:
            cv2.imshow("results", stacked_result_rgb)
            cv2.waitKey(1)
        return scan

    def create_filled_masks(self, results: Results, scan: LidarScan):
        instance_ids = np.empty(0, np.uint32)  # Keep track of which instances are kept
        class_ids = np.empty(0, np.uint32)  # Keep track of which classes are kept
        if results.boxes.id is not None and results.masks is not None:
            mask_edges = results.masks.xy
            orig_instance_ids = np.uint32(results.boxes.id.int())
            orig_class_ids = np.uint32(results.boxes.cls.int())
            # opencv drawContours requires 3-channel float32 image. We'll convert back to uint32 at the end
            instance_id_img = np.zeros((scan.h, scan.w, 3), np.float32)
            # Process ids in reverse order to ensure older instances overwrite newer ones in case of overlap
            for edge, instance_id, class_id in zip(mask_edges[::-1], orig_instance_ids[::-1], orig_class_ids[::-1]):
                if len(edge) != 0:  # It is possible to have an instance with zero edge length. Error check this case
                    instance_id_img = cv2.drawContours(instance_id_img, [np.int32([edge])], -1, color=[np.float64(instance_id), 0, 0], thickness=-1)
                    instance_ids = np.append(instance_ids, instance_id)
                    class_ids = np.append(class_ids, class_id)
            instance_id_img = instance_id_img[..., 0].astype(np.uint32)  # Convert to 1-channel image

            # Remove any instance_ids that were fully overwritten by an overlapping mask
            in_bool = np.isin(instance_ids, instance_id_img)
            instance_ids = instance_ids[in_bool]
            class_ids = class_ids[in_bool]
        else:
            instance_id_img = np.zeros((scan.h, scan.w), np.uint32)

        # Last step make the class id image using a lookup table from instances to classes
        if instance_ids.size > 0:
            instance_to_class_lut = np.arange(0, np.max(instance_ids) + 1, dtype=np.uint32)
            instance_to_class_lut[instance_ids] = class_ids
            class_id_img = instance_to_class_lut[instance_id_img]
        else:
            class_id_img = np.zeros((scan.h, scan.w), np.uint32)

        return instance_id_img, class_id_img, instance_ids, class_ids


if __name__ == '__main__':
    # parse the command arguments
    parser = argparse.ArgumentParser(prog='sdk yolo demo',
                                     description='Runs a minimal demo of yolo post-processing')
    parser.add_argument('source', type=str, help='Sensor hostname or path to a sensor PCAP or OSF file')
    args = parser.parse_args()

    # Example for displaying results with opencv
    #scans = ScanIterator(open_source(args.source, sensor_idx=0, cycle=True), use_opencv=True)
    #for i, scan in enumerate(scans):
    #    if i > 10:  # break after N frames
    #        break

    # Example for displaying results with SimpleViz
    scans = ScanIterator(open_source(args.source, sensor_idx=0, cycle=True), use_opencv=False)
    SimpleViz(scans._metadata, rate=0).run(scans)

