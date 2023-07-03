import json
import numpy as np
import argparse
import os
"""
convert cameras.json into the format of eg3d label
"""
# fname = '/apdcephfs_cq2/share_1290939/kitbai/PTI/data/nerface/1/cropped_images/cameras.json'
parser = argparse.ArgumentParser()
parser.add_argument('--in_root', type=str, default="", help='process folder')
args = parser.parse_args()
in_root = args.in_root

fname = os.path.join(in_root, "cropped_images/cameras.json")
# fname = '/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/datasets/ad_dataset/french/train/cropped_images/cameras.json'
# fname = '/apdcephfs_cq2/share_1290939/kitbai/celeba_sub/images1/cropped_images/cameras.json'
with open(fname, 'rb') as f:
    labels = json.load(f)

results_new = []
for ind in labels.keys():
    pose = np.array(labels[ind]["pose"]).reshape(16)
    pose = list(pose) + list([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1])
    results_new.append((ind, pose))

# with open("/apdcephfs_cq2/share_1290939/kitbai/PTI/data/nerface/1/cropped_images/test.json", 'w') as outfile:
# with open("/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/datasets/ad_dataset/french/train/cropped_images/test.json", 'w') as outfile:
#             json.dump({"labels": results_new}, outfile, indent="\t")
with open(os.path.join(in_root, "cropped_images/test.json"), 'w') as outfile:
            json.dump({"labels": results_new}, outfile, indent="\t")
