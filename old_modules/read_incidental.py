from collections import defaultdict
from natsort import natsorted

import matplotlib.pyplot as plt
import pydicom as dicom
import numpy as np
import os

def read_slices(slices):
    '''
    Read images and other meta_infos from slices list
    :param slices: list of dicom slices
    :return: image in HU and other meta_infos
    '''
    # Sort slices according to the instance number
    slices.sort(key=lambda x: int(x.InstanceNumber))
    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)

    # Read some scan properties
    sliceThickness = slices[0].SliceThickness
    pixelSpacing = slices[0].PixelSpacing
    scanID = slices[0].StudyInstanceUID

    return image, sliceThickness, pixelSpacing, scanID

def add_scan(patientID, date, series, imgPath, sliceThickness, pixelSpacing, scanID, **kwargs):
    '''
    Add current scan meta information into global list
    :param: meta informations for current scan
    :return: scan_info (in dictionary)
    '''
    scanInfo = {
        "patientID": patientID,
        "scanID": scanID,
        "date": date,
        "series": series,
        "imagePath": imgPath,
        "sliceThickness": sliceThickness,
        "pixelSpacing": pixelSpacing,
    }
    scanInfo.update(kwargs)
    return scanInfo


def sample_stack(stack, rows=6, cols=6, start_with=8, show_every=2):
    '''
    Sample slices from CT scan and show
    :param stack: slices of CT scan
    :param rows: rows
    :param cols: cols
    :param start_with: first slice number
    :param show_every: show interval
    :return: none
    '''
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()


rootFolder = "/Users/yuan_pengyu/Downloads/IncidentalLungCTs_sample/"
CTlist = [i for i in os.listdir(rootFolder) if os.path.isdir(os.path.join(rootFolder, i))]
CTlist = natsorted(CTlist)
CTinfo = []
no_CTscans = []
matchMoreThanOne = []
matches = ["LUNG", "lung"]


for CTscanId in CTlist:
    imgFolder = os.path.join(rootFolder, CTscanId, "CT_data")
    sliceList = natsorted(os.listdir(imgFolder))
    # Distribute all slices to different series
    seriesDict = defaultdict(list)
    for sliceID in sliceList:
        sliceDicom = dicom.read_file(os.path.join(imgFolder, sliceID))
        series = sliceDicom.SeriesDescription
        seriesDict[series].append(sliceDicom)
    patientID = sliceDicom.PatientID
    date = sliceDicom.ContentDate
    print("\n>>>>>>> Load patient {:s} at date {:s}".format(patientID, date))
    print("All series types: ", list(seriesDict.keys()))
    lungSeries = [i for i in list(seriesDict.keys()) if np.any([m in i for m in matches])]

    # Process only lung scans
    if len(lungSeries) == 0:
        print("No lung scans found!")
        no_CTscans.append(seriesDict)
    else:
        # assert len(lungSeries) == 1, "More than 1 lung scans found!"
        if len(lungSeries) > 1:
            print("More than 1 lung scans found!")
            id = np.argmin([len(i) for i in lungSeries])
            series = lungSeries[id]
            matchMoreThanOne.append(lungSeries)
        else:
            series = lungSeries[0]
        print("Lung series: ", series)
        slices = seriesDict[series]
        image, sliceThickness, pixelSpacing, scanID = read_slices(slices)
        imagePath = os.path.join(rootFolder, CTscanId, "{:s}-{:s}.npz".format(patientID, date))
        scanInfo = add_scan(patientID, date, series, imagePath, sliceThickness, pixelSpacing, scanID)
        CTinfo.append(scanInfo)
        np.savez_compressed(imagePath, image=image, info=scanInfo)
        print("Save scan to {:s}".format(imagePath))

print("-" * 30 + " CTinfo " + "-" * 30)
[print(i) for i in CTinfo]
CTinfoPath = os.path.join(rootFolder, "CTinfo.npz")
np.savez_compressed(CTinfoPath, list=CTinfo)

print("-" * 30 + " no_CTscans " + "-" * 30)
[print(i) for i in [list(i.keys) for i in no_CTscans]]

print("-" * 30 + " matchMoreThanOne " + "-" * 30)
[print(i) for i in matchMoreThanOne]
