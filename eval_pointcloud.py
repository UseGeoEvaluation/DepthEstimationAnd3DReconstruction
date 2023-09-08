import open3d as o3d
import numpy as np
import sys
from scipy.spatial import KDTree
import argparse
import os
from matplotlib import cm

####################################################################################################
# Calculate completeness


def calculate_completeness_for_error_threshold(source_points: np.array, target_points: np.array, error_threshold: float) -> float:
    # switch target_points and source_points
    correspondence_set_full, _ = find_correspondence_set_with_completeness_threshold(
        target_points, source_points, 1)
    l1, _, _ = eval(target_points[correspondence_set_full[:, 0]],
                    source_points[correspondence_set_full[:, 1]], print_results=False)
    return len(l1[l1 <= error_threshold]) / len(target_points)

####################################################################################################
# Find correspondence set of source and target points with a given completeness threshold


def find_correspondence_set_with_completeness_threshold(source_points: np.array, target_points: np.array, completeness_threshold: float) -> (np.array, float):
    correspondence_set = None

    kdtree = KDTree(target_points, leafsize=20)

    source_points_i = np.arange(len(source_points))
    source_points_i = np.expand_dims(source_points_i, axis=-1)

    step_size = 2
    current_threshold = 1
    if completeness_threshold == 1:
        current_threshold = sys.maxsize
    completeness = 0

    d, i = kdtree.query(source_points, workers=-1)

    i = np.expand_dims(i, axis=-1)
    correspondence_set_raw = np.concatenate([source_points_i, i], axis=-1)
    d = np.squeeze(d)

    # perform search in KDTree to find distance for completeness_threshold
    runs = 0
    while True:
        correspondence_set = correspondence_set_raw[d <= current_threshold]
        completeness = int(len(correspondence_set) /
                           len(source_points) * 10000) / 10000
        if completeness == completeness_threshold or completeness_threshold == 1:
            break

        # if we overshoot the target completeness we half the step size and revert direction of search
        if completeness > completeness_threshold and step_size > 0:
            step_size = - np.abs(step_size / 2)
        elif completeness < completeness_threshold and step_size < 0:
            step_size = np.abs(step_size / 2)

        current_threshold += step_size

        # sanity check
        if runs > 1000:
            print(
                "INFO: Max iterations reached find_correspondence_set_with_completeness_threshold")
            break

        runs += 1

    return correspondence_set, current_threshold

####################################################################################################
# Calculate and print results. returns point to plane error metric


def eval(predictions: np.array, targets: np.array, print_results: bool = True) -> (float, float, float):
    l1 = point_to_point_distance(predictions, targets)
    mse = (l1 ** 2)
    rmse = np.sqrt(mse.mean())

    if print_results:
        print("point_to_point:")
        print("abs: ", l1.mean())
        print("mse: ", mse.mean())
        print("rmse: ", rmse)

    return l1, mse, rmse

####################################################################################################
# Calculate point to point distance


def point_to_point_distance(predictions: np.array, targets: np.array) -> np.array:
    return np.abs(np.linalg.norm(predictions - targets, axis=1))

####################################################################################################
# Perform point to plane icp for alignment and return 4x4 transformation


def point_to_plane_icp(source: np.array, target: np.array, threshold: float, trans_init: np.array) -> np.array:
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-08, relative_rmse=1e-08, max_iteration=1000))
    return reg_p2l.transformation

####################################################################################################
# Perform point to point icp for alignment and return 4x4 transformation


def point_to_point_icp(source: np.array, target: np.array, threshold: float, trans_init: np.array) -> np.array:
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-08, relative_rmse=1e-08, max_iteration=1000))
    return reg_p2l.transformation

####################################################################################################
# Perform point to point icp with scaling for alignment and return 4x4 transformation


def point_to_point_icp_scaling(source: np.array, target: np.array, threshold: float, trans_init: np.array) -> np.array:
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-08, relative_rmse=1e-08, max_iteration=1000))
    return reg_p2l.transformation


####################################################################################################
# Clips the error to a specific range and maps the scaled error to a color
def colorize_by_error(error: np.array, error_range: list = [0, 1]) -> np.array:
    error = np.clip(error, a_min=error_range[0], a_max=error_range[1])
    mn, mx = error_range[0], error_range[1]
    error = (error - mn) / (mx - mn)
    return cm.turbo(error)[:, :3]

####################################################################################################
# Read 4x4 transformation matrix. Syntax: 1 0 0 0
#                                         0 1 0 0
#                                         0 0 1 0
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
# Print text header


def printHeader(estPath: str, gtPath: str, completeness_threshold: float, abs_error_threshold: float, perform_icp_alignment: bool, visualize_abs_error: bool, visualize_destination_path: str, path_to_transformation_matrix: str) -> None:
    print(f'################################################################################')
    print(f'# > Estimate: {estPath}')
    print(f'# > Ground truth: {gtPath}')
    print(f'# > Completeness threshold: {completeness_threshold}')
    print(f'# > Absolute error threshold: {abs_error_threshold}')
    print(f'# > With ICP adjustment: {perform_icp_alignment}')
    if visualize_abs_error:
        print(f'# > Visualize error: {visualize_abs_error}')
        print(
            f'# > Destination path for the color coded point cloud: {visualize_destination_path}')
    if path_to_transformation_matrix:
        print(
            f'# > Path to transformation matrix: {path_to_transformation_matrix}')

    print(f'# -------------------------------------------------------------------------------')


####################################################################################################
# Main entry point into application
if __name__ == "__main__":

    # initialize argument parser
    argParser = argparse.ArgumentParser(
        description="Utility script to evaluate point clouds.")
    argParser.add_argument("-icp", "--perform_icp_alignment", action='store_true',
                           help="Option to specify wether to use icp for better alignment. "
                           "The point clouds must already be at least roughly aligned for this to work.")

    argParser.add_argument("-vis", "--visualize_abs_error", action='store_true',
                           help="Option to visualize the absolut error color coded. "
                           "Saves a color coded point cloud.")

    argParser.add_argument("-vis_dest", "--visualize_destination_path", type=str, default="vis.ply",
                           help="Destination path for the color coded point cloud.")

    argParser.add_argument("-transform", "--path_to_transformation_matrix", type=str, default=None,
                           help="Destination path for a 4x4 matrix to transform the estimated point cloud.")

    argParser.add_argument("-cpl", "--completeness_threshold", type=float, default=0.9,
                           help="Completeness threshold used for calculating the error metrics. "
                           "For the final metrics only the best X% best values are considered. ")

    argParser.add_argument("-abs", "--abs_error_threshold", type=float, default=0.2,
                           help="Error Threshold used to calculate completeness. "
                           "All data points <  abs_error_threshold are treated as inliers.")

    argParser.add_argument(
        "estimate", help="Path to the estimate(s) that is/are to be evaluated.")

    argParser.add_argument(
        "ground_truth", help="Path to the data used as ground.")
    args = argParser.parse_args()

    # get input paths
    estPath = args.estimate
    gtPath = args.ground_truth

    # get optional args
    completeness_threshold = args.completeness_threshold
    abs_error_threshold = args.abs_error_threshold
    perform_icp_alignment = args.perform_icp_alignment
    visualize_abs_error = args.visualize_abs_error
    visualize_destination_path = args.visualize_destination_path
    path_to_transformation_matrix = args.path_to_transformation_matrix

    printHeader(estPath, gtPath, completeness_threshold, abs_error_threshold, perform_icp_alignment,
                visualize_abs_error, visualize_destination_path, path_to_transformation_matrix)

    target_pcd = o3d.io.read_point_cloud(gtPath)
    translate = -np.asarray(target_pcd.get_center())
    target_pcd = target_pcd.translate(translate)
    target_points = np.asarray(target_pcd.points)
    if len(target_points) == 0:
        raise("ERROR: Ground truth point cloud empty")

    source_pcd = o3d.io.read_point_cloud(estPath)

    print("Number of points:", len(np.asarray(source_pcd.points)))
    if path_to_transformation_matrix:
        T = read_transformation_matrix(path_to_transformation_matrix)
        source_pcd = source_pcd.transform(T)

    source_pcd = source_pcd.translate(translate)
    source_points = np.asarray(source_pcd.points)
    if len(source_points) == 0:
        raise("ERROR: Estimate point cloud empty")

    # get correspondence set for given completeness threshold
    correspondence_set, threshold = find_correspondence_set_with_completeness_threshold(
        source_points, target_points, completeness_threshold)

    print("--------- Error without icp alignment and completeness threshold: ",
          completeness_threshold)
    l1, mse, rmse = eval(
        source_points[correspondence_set[:, 0]], target_points[correspondence_set[:, 1]])

    completeness = calculate_completeness_for_error_threshold(
        source_points, target_points, abs_error_threshold)
    print("completeness with l1", abs_error_threshold, ":", completeness)

    if perform_icp_alignment:
        transform_icp = point_to_point_icp(
            source_pcd, target_pcd, threshold, np.eye(4))
        source_pcd = source_pcd.transform(transform_icp)
        source_points = np.asarray(source_pcd.points)

        correspondence_set, threshold = find_correspondence_set_with_completeness_threshold(
            source_points, target_points, completeness_threshold)

        print("--------- Error after icp alignment and completeness threshold: ",
              completeness_threshold)
        l1, mse, rmse = eval(
            source_points[correspondence_set[:, 0]], target_points[correspondence_set[:, 1]])

        completeness = calculate_completeness_for_error_threshold(
            source_points, target_points, abs_error_threshold)
        print("completeness with l1", abs_error_threshold, ":", completeness)

        transform_icp = point_to_point_icp_scaling(
            source_pcd, target_pcd, threshold, np.eye(4))
        source_pcd = source_pcd.transform(transform_icp)
        source_points = np.asarray(source_pcd.points)

        correspondence_set, threshold = find_correspondence_set_with_completeness_threshold(
            source_points, target_points, completeness_threshold)

        print("--------- Error after icp alignment with scaling and completeness threshold: ",
              completeness_threshold)
        l1, mse, rmse = eval(
            source_points[correspondence_set[:, 0]], target_points[correspondence_set[:, 1]])

        completeness = calculate_completeness_for_error_threshold(
            source_points, target_points, abs_error_threshold)
        print("completeness with l1", abs_error_threshold, ":", completeness)

    if visualize_abs_error:
        correspondence_set, threshold = find_correspondence_set_with_completeness_threshold(
            source_points, target_points, 1)
        l1, _, _ = eval(source_points[correspondence_set[:, 0]],
                        target_points[correspondence_set[:, 1]], print_results=False)
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(
            source_points[correspondence_set[:, 0]])
        colors = colorize_by_error(l1, error_range=[0, 0.5])
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(
            visualize_destination_path, colored_pcd.translate(-translate))
        o3d.io.write_point_cloud(visualize_destination_path.replace(
            ".ply", "_rgb.ply"), source_pcd.translate(-translate))
