import numpy as np
import math
from scipy.interpolate import griddata

def calc_theta_degree_in_z(point):
    [x,y,z] = point
    theta_radian = math.atan(z / math.sqrt(x * x + y * y))
    theta = theta_radian * 180 / np.pi
    if (theta >= 2):
        # print("Theta > 2, theta = ", theta)
        theta = 2
    if (theta <= -24.5):
        # print("Theta <= -24.5, theta = ", theta)
        theta = -24.5+0.001
    return theta


def calc_theta_degree_in_xy(point):
    [x,y,z] = point
    if(x != 0):
        theta_s = math.atan(abs(y / x)) * 180 / np.pi
        if x > 0 and y > 0:
            theta = theta_s
        elif x < 0 and y > 0:
            theta = 180 - theta_s
        elif x < 0 and y < 0:
            theta = theta_s + 180
        elif x > 0 and y < 0:
            theta = 360 - theta_s
        elif y == 0 and x > 0:
            theta = 0
        elif y == 0 and x < 0:
            theta = 180
    else:
        if y > 0:
            theta = 90
        else:
            theta = 270
    return theta


def pointcloud_to_rangeimage(pointcloud, lidar_angular_xy_range, max_lidar_angular_z, min_lidar_angular_z, range_x, range_y):
    range_image_double = np.zeros((range_x, range_y))
    length = pointcloud.shape[0]
    for i in range(length):
        point_world = pointcloud[i]
        [x_world, y_world, z_world] = point_world
        degree_in_xy = calc_theta_degree_in_xy(point_world)
        y_rangeimage = math.floor(degree_in_xy / lidar_angular_xy_range * range_y)
        degree_in_z = calc_theta_degree_in_z(point_world)
        x_rangeimage = math.floor((max_lidar_angular_z - degree_in_z) / (max_lidar_angular_z - min_lidar_angular_z) * range_x)

        depth = math.sqrt(x_world * x_world + y_world * y_world + z_world * z_world)
        if depth < range_image_double[x_rangeimage][y_rangeimage] or range_image_double[x_rangeimage][y_rangeimage] == 0:
            range_image_double[x_rangeimage][y_rangeimage] = depth

    return range_image_double


def remove_balck_line_and_remote_points(range_image):
    range_x = range_image.shape[0]
    range_y = range_image.shape[1]
    # initialize
    range_image_completion = range_image
    for height in range(1, range_x-1):
        for width in range(1, range_y-1):
            left = max(width - 1, 0)
            right = min(width + 1, range_y)
            up = max(height - 1, 0)
            down = min(height + 1, range_x)
            if range_image[height, width] == 0:
                # remove straight line
                if range_image[up][width] > 0 and range_image[down][width] > 0 and (range_image[height][left] == 0 or range_image[height][right] == 0):
                    range_image_completion[height][width] = (range_image[up][width] + range_image[down][width]) / 2

    for height in range(1, range_x-1):
        for width in range(1, range_y-1):
            left = max(width - 1, 0)
            right = min(width + 1, range_y)
            up = max(height - 1, 0)
            down = min(height + 1, range_x)
            if range_image_completion[height][width] == 0:
                point_up = range_image_completion[up][width]
                point_down = range_image_completion[down][width]
                point_left = range_image_completion[height][left]
                point_right = range_image_completion[height][right]
                point_left_up = range_image_completion[up][left]
                point_right_up = range_image_completion[up][right]
                point_left_down = range_image_completion[down][left]
                point_right_down = range_image_completion[down][right]
                surround_points = int(point_up != 0) + int(point_down != 0) + int(point_left != 0) + int(
                    point_right != 0) + int(point_left_up != 0) + int(point_right_up != 0) + int(
                    point_left_down != 0) + int(point_right_down != 0)
                if surround_points >= 7:
                    surround_points_sum = point_up + point_down + point_left + point_right + point_left_up + point_right_up + point_left_down + point_right_down
                    range_image_completion[height][width] = surround_points_sum / surround_points

    return range_image_completion

# def image_interpolation(range_image, interpolation_method, range_x, range_y):
