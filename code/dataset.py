import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import json

from scipy.spatial.transform import Rotation as R

import numpy as np
import torch
import torchvision.transforms as transforms

from scipy.ndimage import gaussian_filter1d

from ray_utils import *



def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        #  pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)
        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class HeadData(Dataset):
    def __init__(self, split, transform=None, dataset = 'nerface', person = 'person_2',ds_path=None,suffix = '.png' ):

        if split == 'train':
            self.ds_path = './code/datasets/'+ dataset +'/'+person +'/train/cropped_images'
        elif split == 'test':
            self.ds_path = './code/datasets/'+ dataset +'/'+person +'/test2/cropped_images'
        else:
            self.ds_path = ds_path
            # raise NotImplementedError
        label_path = self.ds_path + '/test.json'

        if label_path is not None:
            with open(label_path, 'rb') as f:
                labels = json.load(f)['labels']
            self.labels = dict(labels)

        self.frames = glob.glob(self.ds_path + '/*'+ suffix)
        if not ( split == 'train'):
            self.frames = sorted(self.frames)

        self.transform = transform

    def get_label(self, idx):

        label = [self.labels[idx]]


        label = np.array(label)
        label[:, [1,2,5,6,9,10]] *= -1 # opencv --> opengl, only for orignal eg3d
        label = label.astype({1: np.int64, 2: np.float32}[label.ndim])
        return torch.tensor(label).squeeze(0)

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        label = self.get_label(frame_path.split('/')[-1].split('.')[0]+'.png')

        image = Image.open(frame_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
            return image, label
        
        return image, label

    def __len__(self):
        return len(self.frames)




class HeadData_test(Dataset):
    def __init__(self, split, transform=None, dataset = 'nerface', person = 'person_2',ds_path=None,suffix = '.png' ):

        if split == 'train':
            self.ds_path = './code/datasets/'+ dataset +'/'+person +'/train/cropped_images'
        elif split == 'test':
            self.ds_path = './code/datasets/'+ dataset +'/'+person +'/test2/cropped_images'
        else:
            self.ds_path = ds_path
            # raise NotImplementedError
        # self.videos = os.listdir(self.ds_path)
        label_path = self.ds_path + '/test.json'

        if label_path is not None:
            with open(label_path, 'rb') as f:
                labels = json.load(f)['labels']
            self.labels = dict(labels)

        self.frames = glob.glob(self.ds_path + '/*'+ suffix)
        self.frames = sorted(self.frames)

        self.transform = transform


    def get_soomth_labels(self):
        soomth_labels = []
        for frame in self.frames:

            label = self.labels[frame.split('/')[-1].split('.')[0]+'.png']
            soomth_labels.append(label)
        soomth_labels = np.array(soomth_labels)
        soomth_labels = gaussian_filter1d(soomth_labels, 3, 0)
        re_labels = {}
        for i in range(len(self.frames)):
            re_labels[self.frames[i].split('/')[-1].split('.')[0]+'.png'] = soomth_labels[i,:]
        return re_labels



    def get_label(self, idx):
        label = [self.labels[idx]]

        label = np.array(label)
        label[:, [1,2,5,6,9,10]] *= -1 # opencv --> opengl, only for orignal eg3d
        label = label.astype({1: np.int64, 2: np.float32}[label.ndim])
        return torch.tensor(label).squeeze(0)

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        label = self.get_label(frame_path.split('/')[-1].split('.')[0]+'.png')

        image = Image.open(frame_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
            return image, label
        
        return image, label

    def __len__(self):
        return len(self.frames)


class HeadData_3DMM(Dataset):
    def __init__(self, split, transform=None, dataset = 'nerface', person = 'person_2',ds_path=None ):

        if split == 'train':
            self.ds_path = './code/datasets/'+ dataset +'/'+person +'/train/cropped_images'
        elif split == 'test':
            self.ds_path = './code/datasets/'+ dataset +'/'+person +'/test2/cropped_images'
        else:
            self.ds_path = ds_path
        label_path = self.ds_path + '/test.json'

        if label_path is not None:
            with open(label_path, 'rb') as f:
                labels = json.load(f)['labels']
            self.labels = dict(labels)

        self.frames = glob.glob(self.ds_path + '/*.png')
        if not ( split == 'train'):
            self.frames = sorted(self.frames)
            # self.labels = self.get_soomth_labels()

        self.transform = transform
        with open(os.path.join('./code/datasets/'+ dataset +'/'+person, f"transforms_{split}.json"), "r") as fp:
            metas = json.load(fp)
        self.poses = {}
        self.expressions = {}

        for frame in metas["frames"]:
            fname = frame["file_path"].split('/')[-1] + ".png"
            self.poses[fname] = np.array(frame["transform_matrix"])

            self.expressions[fname] = np.array(frame["expression"])


    def rotate_labels(self):
        rotated_labels = {}
        for frame in self.frames:

            label = self.labels[frame.split('/')[-1].split('.')[0]+'.png']
            matrix = label[:-9].reshape(4, -1)

            r = matrix[:3, :]

            rot = R.from_rotvec([0, 30 * np.pi / 180., 0]) * R.from_rotvec([ 0 * np.pi / 180., 0, 0])

            c = (np.dot(rot.as_matrix(), r))
            
            matrix[:3, :] = c #.as_matrix()

            rotated_label = np.concatenate((matrix.reshape(-1), np.array([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1])), -1)
            rotated_labels[frame.split('/')[-1].split('.')[0]+'.png'] = rotated_label.reshape(-1)

        return rotated_labels

    def get_soomth_labels(self):
        soomth_labels = []
        for frame in self.frames:

            label = self.labels[frame.split('/')[-1].split('.')[0]+'.png']
            soomth_labels.append(label)
        soomth_labels = np.array(soomth_labels)
        soomth_labels = gaussian_filter1d(soomth_labels, 3, 0)
        re_labels = {}
        for i in range(len(self.frames)):
            re_labels[self.frames[i].split('/')[-1].split('.')[0]+'.png'] = soomth_labels[i,:]
        return re_labels

    def get_label(self, idx):
        label = [self.labels[idx]]
        label = np.array(label)
        label[:, [1,2,5,6,9,10]] *= -1 # opencv --> opengl, only for orignal eg3d
        label = label.astype({1: np.int64, 2: np.float32}[label.ndim])
        return torch.tensor(label).squeeze(0)

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        label = self.get_label(frame_path.split('/')[-1])

        image = Image.open(frame_path).convert('RGB')
        params = self.expressions[frame_path.split('/')[-1]].astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)
            return image, label, params
        
        return image, label, params

    def __len__(self):
        return len(self.frames)




class HeadData_Audio(Dataset):
    def __init__(self, split, transform=None, dataset = 'nerface', person = 'person_2', ds_path=None ):

        if split == 'train':
            self.ds_path = './code/datasets/'+ dataset +'/'+person +'/train/cropped_images'
        elif split == 'val':
            self.ds_path = './code/datasets/'+ dataset +'/'+person +'/test/cropped_images'
        else:
            self.ds_path = ds_path
        label_path = self.ds_path + '/test.json'

        if label_path is not None:
            with open(label_path, 'rb') as f:
                labels = json.load(f)['labels']
            self.labels = dict(labels)

        self.frames = glob.glob(self.ds_path + '/*.jpg')
        if not ( split == 'train'):
            self.frames = sorted(self.frames , key=lambda x:int(x.split('/')[-1].split('.')[0]))

        self.transform = transform

        with open(os.path.join('./code/datasets/'+ dataset +'/'+person, f"transforms_{split}.json"), "r") as fp:
            metas = json.load(fp)
        self.poses = {}
        self.audios = {}
        self.aud_features = np.load(os.path.join(self.ds_path.rsplit("/",2)[0], 'aud.npy'))
        for frame in metas["frames"]:
            fname = str(frame["img_id"]) + ".jpg"
            self.poses[fname] = np.array(frame["transform_matrix"])

            self.audios[fname] = self.aud_features[min(frame['aud_id'], self.aud_features.shape[0]-1)] #np.array(frame["aud_id"])


    def rotate_labels(self):
        rotated_labels = {}
        for frame in self.frames:

            label = self.labels[frame.split('/')[-1].split('.')[0]+'.png']
            label = np.array(label)
            matrix = label[:-9].reshape(4, -1)
            r = matrix[:3, :]
            rot = R.from_rotvec([0, -20 * np.pi / 180., 0]) * R.from_rotvec([ 0 * np.pi / 180., 0, 0])
            c = (np.dot(rot.as_matrix(), r))
            
            matrix[:3, :] = c #.as_matrix()
            rotated_label = np.concatenate((matrix.reshape(-1), np.array([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1])), -1)
            rotated_labels[frame.split('/')[-1].split('.')[0]+'.png'] = rotated_label.reshape(-1)

        return rotated_labels


    def get_label(self, idx):
        label = [self.labels[idx]]
        label = np.array(label)
        label[:, [1,2,5,6,9,10]] *= -1 # opencv --> opengl, only for orignal eg3d
        label = label.astype({1: np.int64, 2: np.float32}[label.ndim])
        return torch.tensor(label).squeeze(0)

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        # print(frame_path)
        label = self.get_label(frame_path.split('/')[-1].split('.')[0] + '.png')
        img_i = torch.zeros(1).long() + int(frame_path.split('/')[-1].split('.')[0])

        image = Image.open(frame_path).convert('RGB')
        audio = self.audios[frame_path.split('/')[-1]].astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)
            return image, label, audio, img_i
        
        return image, label, audio, img_i

    def __len__(self):
        return len(self.frames)

