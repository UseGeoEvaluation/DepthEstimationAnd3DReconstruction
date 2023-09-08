from fileinput import filename
from typing import List
import os
import glob
import argparse
import numpy as np
import cv2

# theta threshold (in m) for absolute error computation
THETA_ABS = [1.0, 0.5, 0.25, 0.10, 0.05, 0.01]

# theta threshold for relative error computation
THETA_REL = [1.25, 1.20, 1.15, 1.10, 1.05, 1.01]

####################################################################################################
#


def applyMedianScaling(estMap: np.array, gtMap: np.array) -> np.array:

    # compute mask of valid pixels
    mask = np.logical_and(estMap > 0, gtMap > 0)

    # compute scale factor and apply
    scaleFct = np.median(gtMap[mask]) / np.median(estMap[mask])
    estMap[mask] *= scaleFct

    return estMap

####################################################################################################
#


def computeAbsoluteError(estMap: np.array, gtMap: np.array) -> List[float]:
    retVal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # count pixels
    pxCtrUnion = np.count_nonzero(estMap * gtMap)
    pxCtrEst = np.count_nonzero(estMap)
    pxCtrGt = np.count_nonzero(gtMap)

    if pxCtrUnion == 0 or pxCtrEst == 0 or pxCtrGt == 0:
        print("ZeroDivisionError: division by zero",
              pxCtrUnion, pxCtrEst, pxCtrGt)
        return retVal

    # compute mask of valid pixels
    mask = np.logical_and(estMap > 0, gtMap > 0)

    # abs-l1
    retVal[0] = np.sum(np.abs(estMap[mask] - gtMap[mask])) / pxCtrUnion

    # Acc-Cpl
    abs_l1 = np.abs(estMap[mask] - gtMap[mask])
    for k in range(0, len(THETA_ABS)):
        validCtr = np.count_nonzero((abs_l1 < THETA_ABS[k]))
        retVal[1 + k * 2] = validCtr / pxCtrEst
        retVal[2 + k * 2] = validCtr / pxCtrGt

    return retVal


####################################################################################################
#
def computeRelativeError(estMap: np.array, gtMap: np.array) -> List[float]:
    retVal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # count pixels
    pxCtrUnion = np.count_nonzero(estMap * gtMap)
    pxCtrEst = np.count_nonzero(estMap)
    pxCtrGt = np.count_nonzero(gtMap)

    if pxCtrUnion == 0 or pxCtrEst == 0 or pxCtrGt == 0:
        print("ZeroDivisionError: division by zero",
              pxCtrUnion, pxCtrEst, pxCtrGt)
        return retVal

    # compute mask of valid pixels
    mask = np.logical_and(estMap > 0, gtMap > 0)

    # rel-l1
    retVal[0] = np.sum(np.abs(estMap[mask] - gtMap[mask]) /
                       gtMap[mask]) / pxCtrUnion

    # Acc-Cpl
    ratioMap = np.maximum((estMap[mask]/gtMap[mask]),
                          (gtMap[mask]/estMap[mask]))
    for k in range(0, len(THETA_REL)):
        validCtr = np.count_nonzero((ratioMap < THETA_REL[k]))
        retVal[1 + k * 2] = validCtr / pxCtrEst
        retVal[2 + k * 2] = validCtr / pxCtrGt

    return retVal

####################################################################################################
#


def processSingleFile(estPath: str, gtPath: str, isEstNotScaled: bool, useAbsError: bool) -> (str, List[float]):

    # read image data
    estMap = cv2.imread(estPath, cv2.IMREAD_UNCHANGED)
    gtMap = cv2.imread(gtPath, cv2.IMREAD_UNCHANGED)

    # get size of arrays
    estSize = estMap.shape
    gtSize = gtMap.shape

    # check if arrays are of same size
    if(estSize != gtSize):
        estMap = cv2.resize(
            estMap, (gtSize[1], gtSize[0]), interpolation=cv2.INTER_NEAREST)

    # set all negative (i.e. invalid) pixel to zero
    estMap.clip(min=0)
    gtMap.clip(min=0)

    # get name from file path
    filename = os.path.basename(estPath)

    # use median scaling for alignment
    if(isEstNotScaled):
        estMap = applyMedianScaling(estMap, gtMap)

    # compute Error
    if useAbsError:
        return filename, computeAbsoluteError(estMap, gtMap)
    else:
        return filename, computeRelativeError(estMap, gtMap)


####################################################################################################
#
def processDirectory(estPath: str, gtPath: str, isEstNotScaled: bool, useAbsError: bool) -> (str, List[float]):
    filenameList = list()
    errorList = list()

    # get files in estimate folder
    estimateFileList = glob.glob(os.path.join(estPath, "*.tiff"))

    cntr = 1
    for estimateFilePath in estimateFileList:
        print(
            f'> Processing file {cntr} / {len(estimateFileList)}', end='\r', flush=True)

        gtFilePath = str(os.path.join(
            gtPath, os.path.basename(estimateFilePath)))

        if os.path.exists(gtFilePath):
            filename, error = processSingleFile(
                estimateFilePath, gtFilePath, isEstNotScaled, useAbsError)
            filenameList.append(filename)
            errorList.append(error)
        else:
            raise ValueError(
                "ERROR: Ground truth file not found: " + gtFilePath)

        cntr += 1
    print('')

    return filenameList, errorList

####################################################################################################
#


def printHeader(estPath: str, gtPath: str, isEstNotScaled: bool, useAbsError: bool, isEstDirectory: bool) -> None:
    print(f'################################################################################')
    if (isEstDirectory):
        print(f'# Processing Directories ')
    else:
        print(f'# Processing Files ')
    print(f'# > Estimate: {estPath}')
    print(f'# > Ground truth: {gtPath}')
    print(f'# > No scale: {isEstNotScaled}')
    print(f'# > Absolute error measure: {useAbsError}')

    print(f'# -------------------------------------------------------------------------------')

####################################################################################################
#


def printResultHeader(useAbsError: bool) -> None:
    print(f'# Filename', end='', flush=True)
    if not useAbsError:
        print(f';L1-rel', end='', flush=True)
        for i in range(0, len(THETA_REL)):
            print(
                f';Acc_{THETA_REL[i]};Cpl_{THETA_REL[i]}', end='', flush=True)
    else:
        print(f';L1-abs', end='', flush=True)
        for i in range(0, len(THETA_ABS)):
            print(
                f';Acc_{THETA_ABS[i]};Cpl_{THETA_ABS[i]}', end='', flush=True)
    print('')

####################################################################################################
#


def printSingleFileResult(filename: str, error: List[float], useAbsError: bool) -> None:
    printResultHeader(useAbsError)
    print(f'{filename}', end='', flush=True)
    for i in range(0, len(error)):
        print(f';{error[i]}', end='', flush=True)
    print('')

####################################################################################################
#


def printDirectoryResult(filenameList: List[str], errorList: List[List[float]], useAbsError: bool) -> None:

    print(f'# -------------------------------------------------------------------------------')
    printResultHeader(useAbsError)

    # initialize error sum for computation of mean value
    errorSum = list()
    for j in range(0, len(errorList[0])):
        errorSum.append(0)

    for i in range(0, len(filenameList)):
        print(f'{filenameList[i]}', end='', flush=True)
        for j in range(0, len(errorList[i])):
            errorSum[j] += errorList[i][j]
            print(f';{errorList[i][j]}', end='', flush=True)
        print('')

    # compute mean and print
    print(f'# -------------------------------------------------------------------------------')
    print(f'Mean', end='', flush=True)
    for j in range(0, len(errorSum)):
        meanVal = errorSum[j] / len(filenameList)
        print(f';{meanVal}', end='', flush=True)
    print('')


####################################################################################################
# Main entry point into application
if __name__ == "__main__":

    # initialize argument parser
    argParser = argparse.ArgumentParser(
        description="Utility script to prepare colmap project to evaluate depth maps.")
    argParser.add_argument("-ns", "--noscale", action='store_true',
                           help="Option to specify that the estimates do not hava a scale.")

    argParser.add_argument("-abs", "--absolute_error", action='store_true',
                           help="Option to specify to compute absolute or relative error measures.")

    argParser.add_argument("estimate", help="Path to the estimate(s) that is/are to be evaluated. "
                                            "This can be a path to a singe .tiff file or a path to "
                                            "a folder holding multiple .tiff files.")

    argParser.add_argument("ground_truth", help="Path to the data used as ground truth depth maps for the "
                                                "evaluation. This can be a path to a singe .tiff "
                                                "file or a path to a folder holding multiple .tiff "
                                                "files.")
    args = argParser.parse_args()

    # get input paths
    estPath = args.estimate
    gtPath = args.ground_truth
    noscale = args.noscale
    absolute_error = args.absolute_error
    # check if input exists
    if not (os.path.exists(estPath)):
        argParser.error(
            f'Input path for estimate does not exist!\nPath: {estPath}')
    if not (os.path.exists(gtPath)):
        argParser.error(
            f'Input path for ground truth does not exist!\nPath: {gtPath}')

    # check if inputs are directories
    if (os.path.isdir(estPath) and os.path.isdir(gtPath)):
        printHeader(estPath, gtPath, noscale, absolute_error, True)
        filenameList, errorList = processDirectory(
            estPath, gtPath, noscale, absolute_error)
        printDirectoryResult(filenameList, errorList, absolute_error)

    # if not check if inputs are single .tiff files
    elif (os.path.splitext(estPath)[1] == ".tiff" and os.path.splitext(gtPath)[1] == ".tiff"):
        printHeader(estPath, gtPath, noscale, absolute_error, False)
        filename, error = processSingleFile(
            estPath, gtPath, noscale, absolute_error)
        printSingleFileResult(filename, error, absolute_error)
    # else print error
    else:
        argParser.error(f'Both input paths should either be directories or single .tiff files!\n'
                        f'Path estimates: {estPath}\n'
                        f'Path ground truth: {gtPath}')
