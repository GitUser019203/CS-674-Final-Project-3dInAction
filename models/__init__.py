import os
import importlib
import sys

from .pointnet import PointNet1, PointNet1Basic
from .pytorch_3dmfv import FourDmFVNet
from .set_transformer import SetTransformerTemporal
from .DGCNN import DGCNN


__all__ = {
    'pn1': PointNet1,
    'pn1_4d_basic': PointNet1Basic,
    '3dmfv': FourDmFVNet,
    'set_transformer': SetTransformerTemporal,
    'dgcnn': DGCNN,
}

def build_model(model_cfg, num_class, frames_per_clip):
    model = __all__[model_cfg['pc_model']](
        model_cfg=model_cfg, num_class=num_class, n_frames=frames_per_clip
    )
    return model

file_name_dict = {
    'pn1': "pointnet.py",
    'pn1_4d_basic': "pointnet.py",
    '3dmfv': "pytorch_3dmfv.py",
    'set_transformer': 'set_transformer.py',
    'dgcnn': 'DGCNN.py',
}


class build_model_from_logdir(object):
    def __init__(self, logdir, model_cfg, num_classes, frames_per_clip):
        pc_model = model_cfg.get('pc_model')
        model_instance = __all__[pc_model]
        model_name = model_instance.__name__
        file_name = file_name_dict.get(pc_model)

        spec = importlib.util.spec_from_file_location(model_name, os.path.join(logdir, 'models', file_name))
        import_model = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = import_model
        spec.loader.exec_module(import_model)
        self.model = model_instance(model_cfg=model_cfg, num_class=num_classes, n_frames=frames_per_clip)
    def get(self):
        return self.model
