import os
import pdb
import torch
import importlib
import torch.nn as nn

from config import system_configs
from models.py_utils.data_parallel import DataParallel
import time
torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, xs, ys, **kwargs):
        # import time
        # t1 = time.clock()
        # print('2',time.clock()-kwargs['time'])
        # t2 =time.clock()
        preds = self.model(*xs, **kwargs)
        # print('3',time.clock()-t2)
        loss_kp  = self.loss(preds, ys, **kwargs)
        # print('l', time.clock() - t2)
        return loss_kp

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, db):
        super(NetworkFactory, self).__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        print("module_file: {}".format(module_file))
        nnet_module = importlib.import_module(module_file)

        self.model   = DummyModule(nnet_module.model(db))
        self.loss    = nnet_module.loss
        self.network = Network(self.model, self.loss)
        self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes).cuda()
        # self.network = torch.nn.DataParallel(self.network,device_ids=system_configs.chunk_sizes)

        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("total parameters: {}".format(total_params))

        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self, xs, ys, **kwargs):
        xs = [x for x in xs]
        ys = [y for y in ys]
        # t2 = time.clock()
        self.optimizer.zero_grad()

        # print('1',time.clock()-t2)
        # t2 = time.clock()
        # print('4', time.clock() - kwargs['time'])
        # kwargs['time']=time.clock()
        loss_kp = self.network(xs, ys,**kwargs)
        # print('1',time.clock()-t2)
        # t2 = time.clock()
        # t2 = time.clock()
        # print('p',t2-t1)
        loss        = loss_kp[0]
        focal_loss  = loss_kp[1]
        sizes_loss   = loss_kp[2]
        offsets_loss   = loss_kp[3]
        loss        = loss.mean()
        focal_loss  = focal_loss.mean()
        sizes_loss   = sizes_loss.mean()
        offsets_loss = offsets_loss.mean()
        loss.backward()
        # print('6', time.clock() - t2)
        # t2 = time.clock()
        self.optimizer.step()
        # print('7', time.clock() - t2)
        return loss, focal_loss, sizes_loss, offsets_loss

    def validate(self, xs, ys, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]

            loss_kp = self.network(xs, ys, **kwargs)
            loss       = loss_kp[0]
            loss = loss.mean()
            return loss

    def test(self, xs, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def save_tensor(self, xs, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]

            kwargs['save'] = True
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            # params = torch.load(f)
            # self.model.load_state_dict(params)
            params = torch.load(f)
            model_dict = self.model.state_dict()
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            self.model.load_state_dict(model_dict)


    def load_params(self, iteration,cache=True):
        if cache:
            cache_file = system_configs.snapshot_file.format(iteration)
        else:
            cache_file = system_configs.snapshot_file.format('cache')
        print("loading model from {}".format(cache_file))
        with open(cache_file, "rb") as f:
            checkpoint = torch.load(f)
            if iteration!=1:
                assert iteration == checkpoint['iteration'],'The iteration is not match'

            model_dict = self.model.state_dict()
            params = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
            model_dict.update(params)
            self.model.load_state_dict(model_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['iteration']

    def save_params(self, iteration,cache=True):
        if cache:
            cache_file = system_configs.snapshot_file.format(iteration)
        else:
            cache_file = system_configs.snapshot_file.format('cache')
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),'iteration':iteration}
            torch.save(params, f)