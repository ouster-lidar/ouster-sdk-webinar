from functools import partial
from ouster.sdk import client
from ouster.sdk import viz, open_source
import numpy as np


class CustomViz:

    def __init__(self, source_path):
        # Open a source and read the RANGE field from the first scan
        source = open_source(source_path)
        self.scan = next(iter(source))
        self.rng = self.scan.field("RANGE")
        self.near_ir = self.scan.field("NEAR_IR")
        self.near_ir_destaggered = client.destagger(source.metadata, self.near_ir)

        # Scale the values from 0 to 1
        self.near_ir = self.near_ir / np.percentile(self.near_ir_destaggered, 95)
        self.near_ir_destaggered = self.near_ir_destaggered / np.percentile(self.near_ir_destaggered, 95)

        # create an XYZLut - a lookup table to efficiently compute
        # Cartesian coordinates from the range field
        xyzlut = client.XYZLut(source.metadata, use_extrinsics=True)
        self.points = client.destagger(source.metadata, xyzlut(self.rng))

        # pause for a moment - what does the "points" object look like?
        # breakpoint()

        # Construct a PointViz - a 3D viewer capable of displaying point clouds
        self.point_viz = viz.PointViz("Hello World!")
        viz.add_default_controls(self.point_viz)

        # Add the points to a Cloud object
        self.highlighted_pixel = viz.Cloud(1)
        self.highlighted_pixel.set_mask([[0, 1, 0, 1]])
        self.highlighted_pixel.set_point_size(10)
        self.point_viz.add(self.highlighted_pixel)
        self.cloud = viz.Cloud(source.metadata)
        self.cloud.set_range(self.rng)
        self.cloud.set_point_size(3)

        # Give the points some color
        self.cloud.set_key(self.near_ir)
        print(dir(viz))
        #self.cloud.set_palette(viz.)

        # Add the Cloud to the viz object and display it
        self.point_viz.add(self.cloud)
        self.point_viz.push_mouse_pos_handler(partial(self.mouse_pos))

        self.image = viz.Image()
        self.image.set_image(self.near_ir_destaggered)
        self.image.set_position(-1, 1, 0.5, 1)
        self.point_viz.add(self.image)

        self.label = viz.Label("", 0, 0)
        self.label.set_scale(3)
        self.label.set_rgba((0, 1, 0, 1))
        self.point_viz.add(self.label)

        self.cuboid_transform = np.eye(4)
        self.cuboid_transform[:3, :3] # *= 0.2
        self.cuboid = viz.Cuboid(self.cuboid_transform, (0, 1, 0, 0.1))
        self.point_viz.add(self.cuboid)

        # Update the viz with all the widgets we added
        self.point_viz.update()

    def mouse_pos(self, ctx, x, y):
        image_pixel = self.image.window_coordinates_to_image_pixel(ctx, x, y)
        if image_pixel:
            try:
                x, y, z = self.points[image_pixel]
                value = self.near_ir_destaggered[image_pixel]
                self.cuboid_transform[:3, 3] = [x, y, z]
                self.highlighted_pixel.set_xyz([[x, y, z]])
                self.cuboid.set_transform(self.cuboid_transform)
                label_text = f"{x:.2f}, {y:.2f}, {z:.2f}\nvalue = {value:.2f}"
                self.label.set_text(label_text)
                self.label.set_position(x, y, z)
                self.point_viz.update()
            except IndexError:
                pass
        return True

    def run(self):
        self.point_viz.run()


if __name__ == '__main__':
    CustomViz('webinar.osf').run()
