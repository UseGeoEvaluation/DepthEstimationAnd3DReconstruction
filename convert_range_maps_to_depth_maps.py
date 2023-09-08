import numpy as np
import cv2
import tifffile as tiff
import open3d as o3d
import glob
import os
import argparse

####################################################################################################
# Main entry point into application
if __name__ == "__main__":

    # initialize argument parser
    argParser = argparse.ArgumentParser(
        description="Utility script to convert range maps to depth maps.")

    argParser.add_argument("ground_truth", help="Path to the data used as ground truth for the "
                                                "evaluation. This can be a path to a singe .tiff "
                                                "file or a path to a folder holding multiple .tiff "
                                                "files.")
    argParser.add_argument("output_dir", help="Path to the output directory.")

    argParser.add_argument("-f", "--focal_length", type=float, default=1159.410657,
                           help="Focal length of the full resolution images.")

    argParser.add_argument("-cx", "--principle_point_x", type=float, default=992.696773,
                           help="Principle point (x) of the full resolution images.")

    argParser.add_argument("-cy", "--principle_point_y", type=float, default=669.486797,
                           help="Principle point (y) of the full resolution images.")

    argParser.add_argument("-vis", "--visualize", action='store_true',
                           help="Option to visualize the depth map.")

    argParser.add_argument("-rgb_dir", "--rgb_dir", type=str,
                           help="Path to the rgb directory ")

    args = argParser.parse_args()

    # get input path
    gtPath = args.ground_truth
    outputPath = args.output_dir
    rgb_dir = args.rgb_dir
    vis = args.visualize

    # check if input exists
    if not (os.path.exists(outputPath)):
        argParser.error(f'Output path for does not exist!\nPath: {outputPath}')
    if not (os.path.exists(gtPath)):
        argParser.error(
            f'Input path for ground truth does not exist!\nPath: {gtPath}')

    # check if input is a directory
    file_paths = None
    if (os.path.isdir(gtPath)):
        file_paths = glob.glob(os.path.join(gtPath, "*.tiff"))
    else:
        file_paths = [gtPath]

    f = args.focal_length
    cx = args.principle_point_x
    cy = args.principle_point_y

    for file_path in file_paths:

        range_map = tiff.imread(file_path)

        # index matrices
        y, x = np.indices(range_map.shape, sparse=False)

        u = x - cx
        v = y - cy

        f_mat = np.ones(range_map.shape) * f
        vp = np.stack((u, v, f_mat))
        vp_norm = np.linalg.norm(vp, ord=2, axis=0)

        e = range_map
        P = e * (vp / vp_norm)

        P = P[2, :, :]
        P = np.float32(P)

        tiff.imwrite(os.path.join(outputPath, os.path.basename(file_path)), P)

        # visualize for sanity check
        if vis:
            depth_map_image = o3d.geometry.Image(P)
            color_raw = cv2.imread(os.path.join(rgb_dir, os.path.basename(
                file_path).replace("_depth_res.tiff", "_res.jpg")))
            color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
            color_image = o3d.geometry.Image(color_raw)

            camera = o3d.camera.PinholeCameraIntrinsic()
            camera.set_intrinsics(
                range_map.shape[1], range_map.shape[0], f, f, cx, cy)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_map_image, depth_scale=100, depth_trunc=10, convert_rgb_to_intensity=False)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, camera)

            # Flip it, otherwise the pointcloud will be upside down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                          [0, 0, -1, 0], [0, 0, 0, 1]])
            o3d.visualization.draw_geometries([pcd])
