from model.model import EncoderDecoder
import torch.nn as nn
import torch
from auxiliary.my_utils import yellow_print
from copy import deepcopy
import os
from auxiliary.visualization import Visualizer

class Tester(object):

    def __init__(self, opt, using_downloaded_weights=False):
        super(Tester, self).__init__()
        self.opt = opt
        self.opt.training_media_path = os.path.join(self.opt.dir_name, "training_media")

        if torch.cuda.is_available():
            self.opt.device = torch.device(f"cuda:{self.opt.multi_gpu[0]}")
        else:
            # Run on CPU
            self.opt.device = torch.device(f"cpu")

        self.network = EncoderDecoder(self.opt)
        self.network = nn.DataParallel(self.network, device_ids=self.opt.multi_gpu)

        if self.opt.reload_model_path != "":
            yellow_print(f"Network weights loaded from  {self.opt.reload_model_path}!")
            # print(self.network.state_dict().keys())
            # print(torch.load(self.opt.reload_model_path).keys())
            if using_downloaded_weights:
                self.network.module.load_state_dict(torch.load(self.opt.reload_model_path, map_location='cuda:0'))
            else:
                self.network.load_state_dict(torch.load(self.opt.reload_model_path, map_location='cuda:0'))

        elif self.opt.reload_decoder_path != "":
            opt = deepcopy(self.opt)
            opt.SVR = False
            network = EncoderDecoder(opt)
            network = nn.DataParallel(network, device_ids=opt.multi_gpu)
            network.load_state_dict(torch.load(opt.reload_decoder_path, map_location='cuda:0'))
            self.network.module.decoder = network.module.decoder
            yellow_print(f"Network Decoder weights loaded from  {self.opt.reload_decoder_path}!")

        else:
            yellow_print("No network weights to reload!")
    
    
    def encode(self, points):
        return self.network.module.encoder(points)
    
    def reconstruct(self, points):
        return self.network(points, train=False)

    def reconstruct_mesh(self, points):
        return self.network.module.generate_mesh(points)
    
    def decode(self, latent_vector):
        return self.network.module.decoder(latent_vector, train=False)
    
    def decode_mesh(self, latent_vector):
        return self.network.module.generate_mesh(latent_vector)
        

if __name__ == '__main__':
    from auxiliary.argument_parser import parser
    opts = parser(['--assemblies','--dir_name','assembly_dataset_scaled_25patches','--nb_primitives','25','--template_type','SQUARE','--reload_model_path','/projects/grail/jamesn8/projects/Shape/AtlasNet/assembly_dataset_scaled_25patches/network.pth'])
    tester = Tester(opts)
    visualizer = Visualizer(8895, opts.env, opts.http_port)
    points = torch.load('/projects/grail/jamesn8/projects/Shape/AtlasNet/dataset/data/cache/AssemblyUnitBallFalse0scale0.5points.pth')

    ind = 2
    pts = points[[ind]]
    outpts = tester.reconstruct(pts.transpose(2, 1))
    mesh = tester.reconstruct_mesh(pts.transpose(2,1).cuda())
    visualizer.show_pointcloud(pts, 'test input')
    if outpts.dim() > 3:
        visualizer.show_pointclouds(outpts,'test output')
    else:
        visualizer.show_pointcloud(outpts, 'test output')
    print('done')