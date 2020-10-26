# Point Clouds visualizer
import mayavi.mlab as mlab
import numpy as np


def downsample_example(fs, downto=16):
    OringLines = 64
    # Read one file from fs and downsample it to $downto line
    data = np.fromfile(fs, dtype=np.float32).reshape((-1,4))
    data_xyz = data[:, :3]
    print(data_xyz.shape)
    # data_xyz_down = data_xyz[:2000]
    # Downsample the sparse point cloud
    velo_down = []
    pt_num = data_xyz.shape[0]
    down_Rate = OringLines / downto
    line_num = int(pt_num / OringLines)

    for i in range(64):
        if i % down_Rate == 0:
            for j in range(int(-line_num/2), int(line_num/2)):
                velo_down.append(data_xyz[i*line_num+j])
    data_xyz_down = np.array(velo_down)

    viz_pts(pts=data_xyz, ref_pts=data_xyz_down)


def get_fig():
    fig = mlab.figure(
        figure=None, bgcolor=(0.4, 0.4, 0.4),
        fgcolor=None, engine=None, size=(500, 500))
    return fig


def viz_pts(pts, ref_pts=None, mode='sphere'):
    # print("Drawing clusters ....")
    fig = mlab.figure(
        figure=None, bgcolor=(0.4, 0.4, 0.4),
        fgcolor=None, engine=None, size=(500, 500))

    color = tuple(np.random.randint(0, 256, size=(1, 3), dtype=np.uint8)[0] / 255.)
    mlab.points3d(
        pts[:, 0], pts[:, 1], pts[:, 2], mode=mode,
        colormap='gnuplot', scale_factor=0.1, figure=fig, color=color)
    if ref_pts is not None:
        color = (0, 1, 0)
        mlab.points3d(
            ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2], mode='arrow',
            colormap='gnuplot', scale_factor=0.5, figure=fig, color=color)
    input("Press any key to continue")


if __name__ == "__main__":
    fs = '/root/kitti/raw/train/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000002.bin'
    downsample_example(fs=fs, downto=16)
