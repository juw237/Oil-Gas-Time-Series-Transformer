import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            #CUDA GPU
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            # device = torch.device('cuda:{}'.format(self.args.gpu))
            #Apple Scilicon GPU
            # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # If Mac GPU (MPS)
            # print('Use GPU: cuda:{}'.format(self.args.gpu))
            if torch.cuda.is_available() and not self.args.use_multi_gpu:
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            elif torch.cuda.is_available() and self.args.use_multi_gpu:
                # For multi-GPU, just setting the environment variable is not enough.
                # You would need to ensure your model and data are properly parallelized across devices.
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                device = torch.device('cuda')
                print(f'Use Multiple GPUs: {self.args.devices}')
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                print('Use GPU: MPS')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
