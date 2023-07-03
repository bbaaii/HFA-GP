"""
    Processes a directory containing *.jpg/png and outputs crops and poses.
"""
import glob
import os
import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='/media/data6/ericryanchan/mafu/Deep3DFaceRecon_pytorch/test_images')
parser.add_argument('--gpu', default=0)
args = parser.parse_args()

print('Processing images:', sorted(glob.glob(os.path.join(args.input_dir, "*"))))

# Compute facial landmarks.
print("Computing facial landmarks for model...")
cmd = "python3.6 /apdcephfs_cq2/share_1290939/kitbai/eg3d-pose-detection-main-master/batch_mtcnn.py"
input_flag = " --in_root " + args.input_dir
cmd += input_flag
# subprocess.run([cmd], shell=True, check=True)
os.system(cmd)


print("Running smooth...")
cmd = "python3.6 /apdcephfs_cq2/share_1290939/kitbai/eg3d-pose-detection-main-master/smooth.py"
input_flag = " --img_folder=" + args.input_dir
cmd += input_flag
os.system(cmd)

# Run model inference to produce crops and raw poses.
print("Running model inference...")
cmd = "python3.6 /apdcephfs_cq2/share_1290939/kitbai/eg3d-pose-detection-main-master/test.py"
input_flag = " --img_folder=" + args.input_dir
gpu_flag = " --gpu_ids=" + str(args.gpu) 
model_name_flag = " --name=face_recon"
model_file_flag = " --epoch=20 "
cmd += input_flag + gpu_flag + model_name_flag + model_file_flag
# subprocess.run([cmd], shell=True, check=True)
os.system(cmd)

# Perform final cropping of 1024x1024 images.
print("Processing final crops...")
cmd = "python3.6 /apdcephfs_cq2/share_1290939/kitbai/eg3d-pose-detection-main-master/crop_images.py"
input_flag = " --indir " + args.input_dir
output_flag = " --outdir " + os.path.join(args.input_dir, 'cropped_images')
cmd += input_flag + output_flag
# subprocess.run([cmd], shell=True, check=True)
os.system(cmd)

# Process poses into our representation -- produces a cameras.json file.
print("Processing final poses...")
cmd = "python3.6 /apdcephfs_cq2/share_1290939/kitbai/eg3d-pose-detection-main-master/3dface2idr.py"
input_flag = " --in_root " + os.path.join(args.input_dir, "epoch_20_000000")
output_flag = " --out_root " + os.path.join(args.input_dir, "cropped_images")

cmd += input_flag + output_flag
# subprocess.run([cmd], shell=True, check=True)
os.system(cmd)

print("Transforming...")
cmd = "python3.6 /apdcephfs_cq2/share_1290939/kitbai/eg3d-pose-detection-main-master/camera2label.py"
input_flag = " --in_root " + args.input_dir
cmd += input_flag
# subprocess.run([cmd], shell=True, check=True)
os.system(cmd)