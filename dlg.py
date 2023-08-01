import torch
from PIL import Image
from torch import nn
from torchvision import transforms, datasets
from utils import dlgUtils
from model import vision
import torch.nn.functional as F


class dlg():
    def __init__(self, images):

        images_transform = []
        tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        for image in images:
            image = Image.open(image)
            images_transform.append(tf(image))
        self.images = torch.stack(images_transform)

        self.dst = datasets.CIFAR100("~/.torch", download=True)
        self.tp = transforms.ToTensor()
        self.tt = transforms.ToPILImage()
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.index = 25

        self.gt_data = self.images.to(self.device)
        gt_label = torch.Tensor([self.dst[self.index + i][1] for i in range(len(self.images))]).long().to(self.device)

        gt_onehot_label = dlgUtils.label_to_onehot(gt_label)

        self.model = vision.LeNet().to(self.device)
        torch.manual_seed(666)
        self.model.apply(vision.weights_init)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # compute original gradient
        pred = self.model(self.gt_data)
        y = self.criterion(pred, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, self.model.parameters())
        self.original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        # generate dummy data and label
        self.dummy_data = torch.randn(self.gt_data.size()).to(self.device).requires_grad_(True)
        self.dummy_label = torch.randn(gt_onehot_label.size()).to(self.device).requires_grad_(True)
        self.optimizer = torch.optim.LBFGS([self.dummy_data, self.dummy_label])

    def run(self):
        tt = transforms.ToPILImage()

        for iters in range(10):
            def closure():
                self.optimizer.zero_grad()

                dummy_pred = self.model(self.dummy_data)
                dummy_onehot_label = F.softmax(self.dummy_label, dim=-1)
                dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, self.original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()

                return grad_diff

            self.optimizer.step(closure)
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())

        return [tt(self.dummy_data[i].cpu()) for i in range(len(self.images))]
