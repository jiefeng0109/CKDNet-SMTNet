import torch.nn as nn
import torch
from .module import Cnn_Module,AEloss
from config import system_configs
class Network(nn.Module):
    def __init__(self,model,loss):
        super(Network,self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, xs, ys):
        preds = self.model(xs)
        loss  = self.loss(preds,ys)

        return loss

class NetworkFactory(object):
    def __init__(self):
        super(NetworkFactory,self).__init__()

        dim = [136, 256,512,128,32]
        self.model = Cnn_Module(dim)
        self.loss  = AEloss()
        self.network = Network(self.model,self.loss).cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.network.parameters())
        )
    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()
    def train(self,xs,ys):
        # xs = [x.cuda() for x in xs]
        ys = [y.cuda() for y in ys]

        self.optimizer.zero_grad()
        loss ,correct,loss1,loss2= self.network(xs,ys)


        loss.backward()
        self.optimizer.step()

        return loss,correct,loss1,loss2

    def validate(self, xs, ys, **kwargs):
        with torch.no_grad():
            # xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]

            loss = self.network(xs, ys)

            return loss

    def test(self, xs, **kwargs):
        with torch.no_grad():
            cla,reg = self.model(xs)
            return torch.nn.functional.softmax(cla,dim=1).cpu().numpy()[:,1],reg.cpu().numpy()

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

    def load_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("loading model from {}".format(cache_file))
        with open(cache_file, "rb") as f:
            params = torch.load(f)
            model_dict = self.model.state_dict()
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            self.model.load_state_dict(model_dict)

    def save_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)

