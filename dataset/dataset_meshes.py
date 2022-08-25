import torch.utils.data as data
import auxiliary.my_utils as my_utils
from .point_sample import sample_point_cloud
import pymesh
import os
from os.path import join, dirname, exists
import pickle
import torch
from copy import deepcopy
import numpy as np

class AssemblyMesh(data.Dataset):

    def __init__(self, opt, train=True):
        self.opt = opt
        self.train = train
        self.num_sample = opt.number_points if train else 2500
        my_utils.red_print('Create Assembly Mesh Dataset...')

        #set up cache folder
        self.path_dataset = join(dirname(__file__), 'data', 'cache')
        if not exists(self.path_dataset):
            os.mkdir(self.path_dataset)
        self.path_dataset = join(self.path_dataset,
                    'Assembly' + self.opt.normalization + str(train) + str(self.opt.max_samples))

        #populate with all mesh paths
        self.datapath = sorted([d.path for d in os.scandir(opt.mesh_path) if d.name.endswith('obj')])
        if self.opt.max_samples > 0:
            self.datapath = self.datapath[:self.opt.max_samples]
        if self.train:
            self.datapath = self.datapath[:int(len(self.datapath) * 0.8)]
        else:
            self.datapath = self.datapath[int(len(self.datapath) * 0.8):]
        
        self.preprocess()
        

    def preprocess(self):
        if exists(self.path_dataset + "info.pkl"):
            # Reload dataset
            my_utils.red_print(f"Reload dataset : {self.path_dataset}")
            with open(self.path_dataset + "info.pkl", "rb") as fp:
                self.data_metadata = pickle.load(fp)

            self.data_points = torch.load(self.path_dataset + "points.pth")
        else:
            # Preprocess dataset and put in cache for future fast reload
            my_utils.red_print("preprocess dataset...")

            # Sample and concatenate all proccessed files

            self.data_points = []
            self.data_metadata = []
            for i in range(self.__len__()):
                mesh, mesh_path = self._getitem(i)
                V = torch.from_numpy(mesh.vertices).float()
                F = torch.from_numpy(mesh.faces).long()
                points, _ = sample_point_cloud(self.num_sample, V, F.T)
                self.data_points.append(points)
                self.data_metadata.append({'mesh_path': mesh_path})
            
            self.data_points = torch.stack(self.data_points)

            # Save in cache
            with open(self.path_dataset + "info.pkl", "wb") as fp:  # Pickling
                pickle.dump(self.data_metadata, fp)
            torch.save(self.data_points, self.path_dataset + "points.pth")


    def _getitem(self, index):
        mesh_path = self.datapath[index]
        mesh = pymesh.meshio.load_mesh(mesh_path)
        return mesh, mesh_path


    
    def __getitem__(self, index):
        return_dict = deepcopy(self.data_metadata[index])
        # Point processing
        points = self.data_points[index]
        points = points.clone()
        return_dict['points'] = points[:, :3].contiguous()
        return return_dict


    def __len__(self):
        return len(self.datapath)