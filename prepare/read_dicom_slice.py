import pydicom as dicom
import numpy as np
import os
import matplotlib.pyplot as plt

output_path = "/home/cougarnet.uh.edu/pyuan2/Downloads/data/Lung_Data/dataset/"
images_path = "/home/cougarnet.uh.edu/pyuan2/Downloads/data/Lung_Data/images/"

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    # slices.sort(key=lambda x: int(x.InstanceNumber))
    slices.sort(key=lambda x: -x.ImagePositionPatient[-1])

    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def shift(coords, patient):
    ImagePosition = patient[0].ImagePositionPatient
    ConstPixelSpacing = (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1], patient[0].SliceThickness)
    x, y, z = zip(*coords)
    x = [(ix - ImagePosition[0]) / ConstPixelSpacing[0] for ix in x]
    y = [(iy - ImagePosition[1]) / ConstPixelSpacing[1] for iy in y]
    # z = [(iz - ImagePosition[2]) / ConstPixelSpacing[2] for iz in z]
    z = [-(iz - ImagePosition[2]) / ConstPixelSpacing[2] for iz in z]
    new_coords = list(zip(*(x, y, z)))
    return new_coords

def get_image(data_path, contour_path, name, save=True):
# data_path = "/home/cougarnet.uh.edu/pyuan2/Downloads/data/Lung_Data/NSCLC-Radiomics/LUNG1-001/09-18-2008-StudyID-69331/0-82046/"
# contour_path = "/home/cougarnet.uh.edu/pyuan2/Downloads/data/Lung_Data/NSCLC-Radiomics/LUNG1-001/09-18-2008-StudyID-69331/0-95085/000000.dcm"
    if save:
        patient = load_scan(data_path)
        if len(patient) == 0:
            return -2
        imgs = get_pixels_hu(patient)
        Contour = dicom.read_file(contour_path)
        try:
            hights = len(Contour.ROIContourSequence[0].ContourSequence)
        except IndexError:
            return -1
        middle = hights // 2
        d = [float(i) for i in Contour.ROIContourSequence[0].ContourSequence[middle].ContourData]
        coords = []
        temp = []
        for i in range(len(d)):
            l = i % 3
            if l == 2:
                temp.append(d[i])
                coords.append(temp)
                temp = []
            else:
                temp.append(d[i])
        new_coords = shift(coords, patient)
        x, y, z = zip(*new_coords)
        slice_id = int(z[0])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(imgs[slice_id], cmap='gray')
        ax.plot(x, y, c='r')
        try:
            fig.savefig(images_path + "/ill/" + "original/" + name + ".png")
        except FileNotFoundError:
            os.makedirs(images_path + "/ill/" + "original/")
            fig.savefig(images_path + "/ill/" + "original/" + name + ".png")
        plt.close(fig)

        x_center = int(np.mean(x))
        y_center = int(np.mean(y))
        z_center = int(np.mean(z))
        width = 64
        sliced_img = imgs[slice_id][y_center - width:y_center + width, x_center - width:x_center + width]
        plt.figure()
        plt.imshow(sliced_img, cmap='gray')
        try:
            plt.savefig(images_path + "/ill/" + "sliced/" + name + ".png")
        except FileNotFoundError:
            os.makedirs(images_path + "/ill/" + "sliced/")
            plt.savefig(images_path + "/ill/" + "sliced/" + name + ".png")
        plt.close()

        try:
            np.save(output_path + name, sliced_img)
        except FileNotFoundError:
            os.makedirs(output_path)
            np.save(output_path + name, sliced_img)

    else:
        file_used = output_path + name + ".npy"
        try:
            sliced_img = np.load(file_used).astype(np.float64)
        except FileNotFoundError:
            return -1

    return sliced_img

def save_image_normal(data_path, name):
    patient = load_scan(data_path)
    if len(patient) == 0:
        print("no data for this patient: ", name)
    else:
        imgs = get_pixels_hu(patient)
        hights = imgs.shape[0]
        slice_id = int(hights // 2)
        plt.figure()
        plt.imshow(imgs[slice_id], cmap='gray')
        try:
            plt.savefig(images_path + "/normal/" + name + ".png")
        except FileNotFoundError:
            os.makedirs(images_path + "/normal/" )
            plt.savefig(images_path + "/normal/" + name + ".png")
        plt.close()

# plt.hist(imgs.flatten(), bins=50, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()


print("")

if __name__ == "__main__":
    from collections import OrderedDict
    from tqdm import tqdm

    root_path = "/home/cougarnet.uh.edu/pyuan2/Downloads/data/Lung_Data/NSCLC-Radiomics"
    mask = False
    Save = True
    ill_people = {}
    normal_people = []

    patient_list = sorted(os.listdir(root_path))
    for i in tqdm(range(len(patient_list))):
        patient = patient_list[i]
        # if int(patient.split('-')[-1]) < 371:
        # if patient != 'LUNG1-361':
        #     continue
        a = os.listdir(root_path + '/' + patient)
        if len(a) == 1:
            folders = os.path.join(root_path + '/' + patient, a[0])
            folders_base = os.listdir(folders)
            if len(folders_base) == 2:
                file = os.listdir(os.path.join(folders, folders_base[0]))
                if len(file) == 1:
                    contour_path = folders + '/' + folders_base[0] + '/' + file[0]
                    data_path = folders + '/' + folders_base[1] + '/' + file[0]
                else:
                    file = os.listdir(os.path.join(folders, folders_base[1]))
                    data_path = os.path.join(folders, folders_base[0])
                    contour_path = folders + '/' + folders_base[1] + '/' + file[0]
                if mask:
                    # sliced_img = slice_with_mask.get_image(data_path, contour_path, patient, Save)
                    pass
                else:
                    sliced_img = get_image(data_path, contour_path, patient, Save)
                if type(sliced_img) == int:
                    if sliced_img == -1:
                        print("missing contour: " + patient + "\n")
                    elif sliced_img == -2:
                        print("no data for this patient: " + patient + "\n")
                else:
                    ill_people[patient] = sliced_img
            else:
                data_path = os.path.join(folders, folders_base[0])
                if Save:
                    if mask:
                        # slice_with_mask.save_image_normal(data_path, patient)
                        pass
                    else:
                        save_image_normal(data_path, patient)
                normal_people.append(patient)
        else:
            print("This file is broken: " + patient + "\n")

    print("finish pre-process the dataset!\n")

    ill_people_sorted = OrderedDict(sorted(ill_people.items(), key=lambda t: int(t[0].split("-")[-1])))
