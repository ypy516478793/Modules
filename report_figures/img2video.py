from natsort import natsorted

import glob
import cv2

img_array = []
for filename in natsorted(glob.glob('/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/sine/cls_5.mbs_25.ubs_10.numstep1.updatelr0.01nonorm.mt70000kp0.90/amp3.00_ph0.00_pts2/*.png')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/sine/cls_5.mbs_25.ubs_10.numstep1.updatelr0.01nonorm.mt70000kp0.90/amp3.00_ph0.00_pts2/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()