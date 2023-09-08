import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import argparse
import os

####################################################################################################
# Read 4x4 transformation matrix. Syntax: 0 0 0 0
#                                         0 0 0 0
#                                         0 0 0 0
#                                         0 0 0 1


def read_transformation_matrix(path_to_transformation_matrix: str) -> np.array:
    if not os.path.isfile(path_to_transformation_matrix):
        raise("ERROR: Transformation matrix not found")
    T = []
    with open(path_to_transformation_matrix, 'r') as f:
        for line in f:
            values = line.strip().split(" ")
            T.append(values)

    return np.array(T).astype(float)

####################################################################################################
# Filter the ground truth by using the distance to the estimated point cloud
# Both are projected onto the XY Plane


def filter_with_correspondence_set(source_points: np.array, target_points: np.array, distance_threshold: float) -> (np.array, float):

    print(target_points[:, :2].shape)
    kdtree = KDTree(target_points[:, :2], leafsize=20)

    source_points_i = np.arange(len(source_points))
    source_points_i = np.expand_dims(source_points_i, axis=-1)

    d, i = kdtree.query(source_points[:, :2], workers=-1)

    i = np.expand_dims(i, axis=-1)
    correspondence_set_raw = np.concatenate([source_points_i, i], axis=-1)
    d = np.squeeze(d)

    inlier = correspondence_set_raw[d <= distance_threshold]
    outlier = correspondence_set_raw[d > distance_threshold]
    return inlier, outlier


####################################################################################################
# Main entry point into application
if __name__ == "__main__":

    # initialize argument parser
    argParser = argparse.ArgumentParser(
        description="Utility script to filter the ground truth point clouds.")
    argParser.add_argument("estimate", help="Path to the estimate.")
    argParser.add_argument(
        "ground_truth", help="Path to the data used as ground.")
    argParser.add_argument("filtered_ground_truth", help="output path")
    argParser.add_argument("-transform", "--path_to_transformation_matrix", type=str, default=None,
                           help="Path for a 4x4 matrix to transform the estimated point cloud.")
    args = argParser.parse_args()

    # get input paths
    estPath = args.estimate
    gtPath = args.ground_truth
    filteredGtPath = args.filtered_ground_truth
    path_to_transformation_matrix = args.path_to_transformation_matrix

    # maximum distance a point of the ground truth point cloud may be away from the estimate to be counted as inlier
    distance_threshold = 5

    estPcd = o3d.io.read_point_cloud(estPath)
    if path_to_transformation_matrix:
        T = read_transformation_matrix(path_to_transformation_matrix)
        estPcd = estPcd.transform(T)
    estPcd_points = np.asarray(estPcd.points)

    gtPcd = o3d.io.read_point_cloud(gtPath)
    gtPcd_points = np.asarray(gtPcd.points)

    inlier, outlier = filter_with_correspondence_set(
        gtPcd_points, estPcd_points, distance_threshold)
    filteredGt_points = gtPcd_points[inlier[:, 0]]
    removedGt_points = gtPcd_points[outlier[:, 0]]

    print(len(filteredGt_points) / len(gtPcd_points))

    filteredPcd = o3d.geometry.PointCloud()
    filteredPcd.points = o3d.utility.Vector3dVector(filteredGt_points)
    o3d.io.write_point_cloud(filteredGtPath, filteredPcd)

    red = np.ones(filteredGt_points.shape) * np.array([0, 0, 1])
    filteredPcd.colors = o3d.utility.Vector3dVector(red)

    outlierPcd = o3d.geometry.PointCloud()
    outlierPcd.points = o3d.utility.Vector3dVector(removedGt_points)
    blue = np.ones(removedGt_points.shape) * np.array([1, 0, 0])
    outlierPcd.colors = o3d.utility.Vector3dVector(blue)

    coloredPcd = filteredPcd + outlierPcd
    o3d.io.write_point_cloud(filteredGtPath.replace(
        ".ply", "_colored.ply"), coloredPcd)
