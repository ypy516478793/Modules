from skimage.measure import regionprops
from skimage.morphology import label
import numpy as np
import colorsys
import random

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def create_masks(bin_mask):
    """
    Create instance masks array based on binary mask
    :param bin_mask: binary mask, shape [None, None]
    :return: instance masks: shape [None, None, num_of_instances]
    """
    labeled_mask = label(bin_mask)
    labeled_mask = cleanEdgedmask(labeled_mask)
    num_instances = len(np.unique(labeled_mask)) - 1
    instance_masks = np.zeros(bin_mask.shape + (num_instances,))
    for i in range(num_instances):
        instance_masks[labeled_mask == i+1, i] = 1
    return instance_masks

def cleanEdgedmask(mask, edgeWidth = 5 ):
    #'''
    #Input: mask (pixel mask array, single cell mask), label Image
    #Output: clean the componnets touching the edge
    #'''
    cleaned_mask = np.copy(mask)
    for obj in regionprops(cleaned_mask*1):
        if mask.shape[0] - obj.bbox [2] < 0 or mask.shape[1] - obj.bbox [3] < 0:
            print ("EROR cleanEdgedmask!!!!!!!!")
        if obj.bbox [0] <=  edgeWidth or \
           obj.bbox [1] <=  edgeWidth or \
           mask.shape[0] - obj.bbox [2]  <=  edgeWidth or \
           mask.shape[1] - obj.bbox [3]  <=  edgeWidth :
           cleaned_mask[cleaned_mask == obj.label] = 0
        #    print ("Edgedmask has been removed ")
    return cleaned_mask

if __name__ == '__main__':
    from skimage.util import img_as_ubyte
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import pickle
    import glob
    import os
    gtfolder = "/home/cougarnet.uh.edu/pyuan2/Datasets/Brain_cell/ground_truth"
    predfolder = "/home/cougarnet.uh.edu/pyuan2/Datasets/Brain_cell/pred_atlas"

    precisions_list, recalls_list, mAP_list = [], [], []
    num_iou_ids = 21
    iou_indices = np.linspace(0, 1, num_iou_ids)
    tplist, fplist, fnlist = [], [], []
    ious_list = []
    for imgpath in tqdm(glob.glob(os.path.join(predfolder, "*_mask.png"))):
        imgid = os.path.basename(imgpath).split("_mask")[0]
        gtpath = os.path.join(gtfolder, imgid + "_mask.png")
        pred = img_as_ubyte(plt.imread(imgpath))
        gt = img_as_ubyte(plt.imread(gtpath))
        pred_masks = create_masks(pred[..., 0]).astype(np.int)  # shape [512, 512, 87]
        gt_masks = create_masks(gt[..., 0]).astype(np.int)  # shape [512, 512, 90]


        overlaps = compute_overlaps_masks(pred_masks, gt_masks)  # shape [87, 90]
        if len(overlaps) == 0:
            print("There is no match in ", imgpath)
            continue
        ids = np.argsort(np.max(overlaps, axis=-1))
        overlaps = overlaps[ids[::-1]]
        # plt.figure()
        # plt.imshow(gt[..., 0])
        # plt.savefig("/home/cougarnet.uh.edu/pyuan2/Datasets/Brain_cell/gt.png", bbox_inches="tight")
        # plt.figure()
        # plt.imshow(label(gt[..., 0]))
        # plt.savefig("/home/cougarnet.uh.edu/pyuan2/Datasets/Brain_cell/gt_instances.png", bbox_inches="tight")

        # Loop through predictions and find matching ground truth boxes
        iou_threshold = 0
        match_count = 0
        pred_match = -1 * np.ones([overlaps.shape[0]])
        gt_match = -1 * np.ones([overlaps.shape[1]])
        for i in range(len(overlaps)):
            # Find best matching ground truth box
            # 1. Sort matches by iou
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low iou
            low_score_idx = np.where(overlaps[i, sorted_ixs] <= iou_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                if gt_match[j] > -1:
                    continue
                # We have a match
                else:
                    match_count += 1
                    gt_match[j] = i
                    pred_match[i] = j
                    break

        precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
        recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

        # Pad with start and end values to simplify the math
        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])
        # Ensure precision values decrease but don't increase. This way, the
        # precision value at each recall threshold is the maximum it can be
        # for all following recall thresholds, as specified by the VOC paper.
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])

        # Compute mean AP over recall range
        indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
        mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                     precisions[indices])

        match_ious = overlaps[np.arange(len(overlaps))[pred_match > -1], pred_match[pred_match > -1].astype(np.int)]
        match_array = match_ious[None, :] >= iou_indices[:, None]
        num_matchs = np.sum(match_array, axis=-1)
        tplist.append(num_matchs)
        fplist.append(np.repeat(np.sum(pred_match == -1), num_iou_ids))
        fnlist.append(len(gt_match) - num_matchs)
        ious_list.append(match_ious)
        precisions_list.append(precisions)
        recalls_list.append(recalls)
        # precisions_list.append(precisions[:-1])
        # recalls_list.append(recalls[:-1])
        mAP_list.append(mAP)

        # plt.plot(recalls, precisions, ".-")
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.title("mAP: {:.3f}".format(mAP))
        # plt.savefig("/home/cougarnet.uh.edu/pyuan2/Datasets/Brain_cell/precision_recall.png", bbox_inches="tight")
        #
        # precision = np.sum(pred_match > -1) / len(pred_match)
        # recall = np.sum(gt_match > -1) / len(gt_match)
        # F1 = 2 * (precision * recall) / (precision + recall)
        #
        # # ids = np.where(pred_match > -1)[0]
        # # ious = overlaps[ids, pred_match[ids].astype(np.int)]
        # # mean_iou = np.mean(ious)
        #
        # ids = np.where(gt_match > -1)[0]
        # ious = overlaps[gt_match[ids].astype(np.int), ids]
        # ious = np.concatenate([ious, np.zeros(np.sum(gt_match == -1))])
        # mean_iou = np.mean(ious)
        #
        # # colors = random_colors(len(np.unique(gt)) - 1)
        # # gt_copy = np.copy(gt)
        # #
        # # for i in range(len(colors)):
        # #     m = np.where(gt[:, :, 0] == i + 1, 1, 0)
        # #     gt_copy = apply_mask(gt_copy, m, colors[i], alpha=1.0)
        # # plt.figure(); plt.imshow(gt_copy)

        print("")

    with open("results.pkl", "wb") as f:
        pickle.dump([tplist, fplist, fnlist, ious_list, precisions_list, recalls_list, mAP_list, iou_indices], f)

    print("")