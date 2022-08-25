import torch
import dataset.dataset_shapenet as dataset_shapenet
import dataset.dataset_meshes as dataset_meshes
import dataset.augmenter as augmenter
from easydict import EasyDict

import dataset.pointcloud_processor as pointcloud_processor
from PIL import Image
import numpy as np

class TrainerDataset(object):
    def __init__(self):
        super(TrainerDataset, self).__init__()
    
    def load(self, path):
        ext = path.split('.')[-1]
        if ext == "npy" or ext == "ply" or ext == "obj":
            return self.load_point_input(path)
        else:
            return self.load_image(path)

    def load_point_input(self, path):
        ext = path.split('.')[-1]
        if ext == "npy":
            points = np.load(path)
        elif ext == "ply" or ext == "obj":
            import pymesh
            points = pymesh.load_mesh(path).vertices
        else:
            print("invalid file extension")

        points = torch.from_numpy(points).float()
        operation = pointcloud_processor.Normalization(points, keep_track=True)
        if self.opt.normalization == "UnitBall":
            operation.normalize_unitL2ball()
        elif self.opt.normalization == "BoundingBox":
            operation.normalize_bounding_box()
        else:
            pass
        return_dict = {
            'points': points,
            'operation': operation,
            'path': path,
        }
        return return_dict


    def load_image(self, path):
        im = Image.open(path)
        im = self.validating(im)
        im = self.transforms(im)
        im = im[:3, :, :]
        return_dict = {
            'image': im.unsqueeze_(0),
            'operation': None,
            'path': path,
        }
        return return_dict

    def build_dataset(self):
        """
        Create dataset
        Author : Thibault Groueix 01.11.2019
        """

        self.datasets = EasyDict()
        # Create Datasets
        if self.opt.assemblies:
            self.datasets.dataset_train = dataset_meshes.AssemblyMesh(self.opt, train=True)
            self.datasets.dataset_test = dataset_meshes.AssemblyMesh(self.opt, train=False)
        else:
            self.datasets.dataset_train = dataset_shapenet.ShapeNet(self.opt, train=True)
            self.datasets.dataset_test = dataset_shapenet.ShapeNet(self.opt, train=False)

        if not self.opt.demo:
            # Create dataloaders
            self.datasets.dataloader_train = torch.utils.data.DataLoader(self.datasets.dataset_train,
                                                                         batch_size=self.opt.batch_size,
                                                                         shuffle=True,
                                                                         num_workers=int(self.opt.workers))
            self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
                                                                        batch_size=self.opt.batch_size_test,
                                                                        shuffle=True, num_workers=int(self.opt.workers))
            axis = []
            if self.opt.data_augmentation_axis_rotation:
                axis = [1]

            flips = []
            if self.opt.data_augmentation_random_flips:
                flips = [0, 2]

            # Create Data Augmentation
            self.datasets.data_augmenter = augmenter.Augmenter(translation=self.opt.random_translation,
                                                               rotation_axis=axis,
                                                               anisotropic_scaling=self.opt.anisotropic_scaling,
                                                               rotation_3D=self.opt.random_rotation,
                                                               flips=flips)

            self.datasets.len_dataset = len(self.datasets.dataset_train)
            self.datasets.len_dataset_test = len(self.datasets.dataset_test)
