import open3d as o3d
import numpy as np
import sys
import pandas as pd
from scipy.spatial import KDTree
import argparse
import os
from matplotlib import cm

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

        # insanity check
        if runs > 1000:
            print(
                "INFO: Max iterations reached find_correspondence_set_with_completeness_threshold")
            break

        runs += 1

    return correspondence_set, current_threshold

####################################################################################################
# Clips the error to a specific range and maps the scaled error to a color


def colorize_by_error(error: np.array, error_range: list = [0, 1]) -> np.array:
    error = np.clip(error, a_min=error_range[0], a_max=error_range[1])
    mn, mx = error_range[0], error_range[1]
    error = (error - mn) / (mx - mn)
    return cm.turbo(error)[:, :3]

####################################################################################################
# Perform point to plane icp for alignment and return 4x4 transformation


def point_to_plane_icp(source: np.array, target: np.array, threshold: float, trans_init: np.array) -> np.array:
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
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
# Perform point to point icp for alignment and return 4x4 transformation


def point_to_point_icp(source: np.array, target: np.array, threshold: float, trans_init: np.array) -> np.array:
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-08, relative_rmse=1e-08, max_iteration=1000))
    return reg_p2l.transformation

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
    argParser = argparse.ArgumentParser(description="Utility script to evaluate meshes with a ground truth point cloud (Estimates normals if no normals are found)."
                                                    "For each triangle, searches for the closest point from the ground truth and calculates the distance.")
    argParser.add_argument("-icp", "--perform_icp_alignment", action='store_true',
                           help="Option to specify wether to use icp for better alignment. "
                           "The point clouds must already be at least roughly aligned for this to work.")

    argParser.add_argument("-vis", "--visualize_abs_error", action='store_true',
                           help="Option to visualize the absolut error color coded. "
                           "Saves a color coded mesh.")

    argParser.add_argument("-vis_dest", "--visualize_destination_path", type=str, default="vis.ply",
                           help="Destination path for the color coded mesh.")

    argParser.add_argument("-transform", "--path_to_transformation_matrix", type=str, default=None,
                           help="Destination path for a 4x4 matrix to transform the estimated mesh.")

    argParser.add_argument("-cpl", "--completeness_threshold", type=float, default=0.9,
                           help="Completeness threshold used for calculating the error metrics. "
                           "For the final metrics only the best X% best values are considered. ")

    argParser.add_argument("-abs", "--abs_error_threshold", type=float, default=0.2,
                           help="Error Threshold used to calculate completeness. "
                           "All data points <  abs_error_threshold are treated as inliers.")

    argParser.add_argument("estimate", help="Path to the estimate(s) that is/are to be evaluated."
                                            "If there are no normal vectors in the point cloud we calculate them with Open3D")

    argParser.add_argument("ground_truth", help="Path to the data used as ground truth for the "
                           "If there are no normal vectors in the point cloud we calculate them with Open3D")
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

    source_mesh = o3d.io.read_triangle_mesh(estPath)
    print("Number of triangles:", len(np.asarray(source_mesh.triangles)))
    if path_to_transformation_matrix:
        T = read_transformation_matrix(path_to_transformation_matrix)
        source_mesh = source_mesh.transform(T)

    target_pcd = o3d.io.read_point_cloud(gtPath)

    # translate to a local coordinate system because of limitations of the raycasting scene
    translate = -np.asarray(target_pcd.get_center())
    target_pcd = target_pcd.translate(translate)
    source_mesh = source_mesh.translate(translate)

    target_points = np.asarray(target_pcd.points)
    target_points = target_points.astype(np.float32)

    if perform_icp_alignment:
        print("--------- Error after icp alignment with scaling and completeness threshold: ",
              completeness_threshold)

        source_points = np.asarray(source_mesh.vertices)

        correspondence_set, threshold = find_correspondence_set_with_completeness_threshold(
            source_points, target_points, completeness_threshold)

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = source_mesh.vertices
        source_pcd.estimate_normals()
        transform_icp = point_to_point_icp(
            source_pcd, target_pcd, threshold, np.eye(4))
        transform_icp = point_to_point_icp_scaling(
            source_pcd, target_pcd, threshold, transform_icp)

        source_mesh = source_mesh.transform(transform_icp)
    else:
        print("--------- Error without icp alignment and completeness threshold: ",
              completeness_threshold)

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(source_mesh)
    initial_mesh_triangle_count = len(mesh_t.triangle.indices)

    distance_to_triangle = np.ones(
        initial_mesh_triangle_count, dtype=float) * -1
    gt_to_triangle_distance = None

    # Iteratively remove triangles from the scene if the closest ground truth point was found. --> one ground truth point could be the closest point for multiple triangles.
    # Repeat until only 1% of initial triangles remain
    while len(mesh_t.triangle.indices[distance_to_triangle == -1]) / initial_mesh_triangle_count > 0.01:

        # Build raycast scene for efficent parallel processing
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(vertex_positions=mesh_t.vertex.positions,
                            triangle_indices=mesh_t.triangle.indices[distance_to_triangle == -1].numpy().astype(np.uint32))

        # Search for the closest point per triangle --> Because the raycast terminates on hit most of the triangles have no corresponding closest point.
        ans = scene.compute_closest_points(target_points)
        point_to_mesh_distances = np.linalg.norm(
            target_points - ans['points'].numpy(), axis=1)

        # Build data frame to group triangle hits by min distance.
        triangle_ids = ans['primitive_ids'].numpy()
        df = pd.DataFrame()
        df["triangle_id"] = triangle_ids
        df["point_to_mesh_distance"] = point_to_mesh_distances

        if gt_to_triangle_distance is None:
            gt_to_triangle_distance = point_to_mesh_distances

        result = df.groupby('triangle_id', as_index=False).min(
            'point_to_mesh_distance')
        min_distance_for_triangle_id = result[[
            "triangle_id", "point_to_mesh_distance"]].to_numpy()

        # Save distances and mark triangles for removal (set to -1)
        triangles_for_iteration = distance_to_triangle[distance_to_triangle == -1]
        triangles_for_iteration[min_distance_for_triangle_id[:, 0].astype(
            int)] = min_distance_for_triangle_id[:, 1]
        distance_to_triangle[distance_to_triangle == -
                             1] = triangles_for_iteration

    distance_to_triangle[distance_to_triangle < 0] = sys.maxsize
    quantile_completeness_threshold = np.quantile(
        distance_to_triangle, completeness_threshold)

    l1 = distance_to_triangle[distance_to_triangle <
                              quantile_completeness_threshold]
    mse = (l1 ** 2)
    rmse = np.sqrt(mse.mean())

    print("abs: ", l1.mean())
    print("mse: ", mse.mean())
    print("rmse: ", rmse)

    completeness = len(
        gt_to_triangle_distance[gt_to_triangle_distance < abs_error_threshold]) / len(target_points)

    print("completeness: ", completeness)

    if visualize_abs_error:

        # open3d uses vertex colors
        # for the visualization, the errors per adjacent triangle are averaged
        vertex_dict = {}

        triangles = np.asarray(source_mesh.triangles)
        for i in range(len(triangles)):
            triangle = triangles[i]
            d = distance_to_triangle[i]
            if d == sys.maxsize:
                continue
            for j in range(3):
                v = triangle[j]
                if v in vertex_dict:
                    vertex_dict[v][0] += 1
                    vertex_dict[v][1] += d
                else:
                    vertex_dict[v] = [1, d]

        vertices = np.asarray(source_mesh.vertices)
        errors = np.ones(len(vertices), dtype=float) * -1
        for i in vertex_dict:
            errors[i] = vertex_dict[i][1] / vertex_dict[i][0]

        colors = np.zeros((len(vertices), 3), dtype=float)
        colors[errors >= 0] = colorize_by_error(
            errors[errors >= 0], error_range=[0, 0.5])

        source_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_triangle_mesh(visualize_destination_path, source_mesh)
