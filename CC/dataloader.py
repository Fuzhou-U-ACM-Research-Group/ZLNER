import re
import torch
from torch.autograd import Variable
from ICCSupervised.ICCSupervised import IDataLoader
from torch.utils.data import TensorDataset, DataLoader, Dataset
from CC.loaders.cn_loader import CNDataLoader
from CC.loaders.cnx_loader import CNXDataLoader
from CC.loaders.le_loader import LLoader
from CC.loaders.lex_loader import LXLoader
from CC.loaders.label_le_loader import LabelLLoader
from CC.loaders.mlabel_le_loader import MLabelLLoader
from CC.loaders.finetune.ftloader_v1 import FTLoaderV1
from CC.loaders.pretrain.ptloader_v1 import PTLoaderV1
from CC.loaders.pretrain.ptloader_v2 import PTLoaderV2
# from CC.loaders.pretrain.ptloader_v3 import PTLoaderV3
from CC.loaders.label_lex_loader import LabelLXLoader
from CC.loaders.le_loader_zl import ZLLoader

class AutoDataLoader(IDataLoader):
    def __init__(self, **args):
        assert "loader_name" in args, "argument loader_name: required loader_name"
        self.loader_name = args['loader_name']
        dataloaders = {
            'cn_loader': CNDataLoader,
            'cnx_loader': CNXDataLoader,
            'le_loader': LLoader,
            'lex_loader': LXLoader,
            'labelle_loader': LabelLLoader,
            'm_labelle_loader': MLabelLLoader,
            'ft_loader_v1': FTLoaderV1,
            'labellex_loader': LabelLXLoader,
            'ptloader_v1': PTLoaderV1,
            'ptloader_v2': PTLoaderV2,
            # 'ptloader_v3': PTLoaderV3,
            'labellex_loader': LabelLXLoader,
            'le_loader_zl':ZLLoader,
        }
        if self.loader_name not in dataloaders:
            raise Exception("Dataloader not found name '{}'.".format(self.loader_name))
        self.loader = dataloaders[self.loader_name](**args)
    
    def __call__(self):
        loader = self.loader
        return loader()