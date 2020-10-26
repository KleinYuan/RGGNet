import numpy as np


def reduce_lidar_line(xyz_intensity, reduce_lidar_line_to):
    origin_lines = 64  # TODO: Remove this hard-coded lines to config
    velo_down = []
    pt_num = xyz_intensity.shape[0]
    down_Rate = origin_lines / reduce_lidar_line_to
    line_num = int(pt_num / origin_lines)

    for i in range(64):
        if i % down_Rate == 0:
            for j in range(int(-line_num/2), int(line_num/2)):
                velo_down.append(xyz_intensity[i*line_num+j])
    data_reduced = np.array(velo_down)
    return data_reduced
