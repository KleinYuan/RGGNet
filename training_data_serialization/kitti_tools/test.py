import numpy as np
from kitti_tools.transform import generate_depth_map, update_depth_map, SE3_to_se3


def test_SE3_to_se3():
    """
    Using online tool as validation: https://www.andre-gaschler.com/rotationconverter/
    """
    # RPY: 1.4, 1.9, 1.5
    example_SE3 = np.array([
        [0.99910773, -0.02616258, 0.03315516, 0.011],
        [0.02697893, 0.99933771, -0.02441879, 0.029],
        [-0.03249434, 0.02529149, 0.99915187, 0.038],
        [0., 0., 0., 1.]])

    se3_get = SE3_to_se3(example_SE3)
    print(se3_get)


test_SE3_to_se3()
