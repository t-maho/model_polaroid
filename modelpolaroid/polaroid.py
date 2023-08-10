import math
import os
import torch
import tqdm
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import torch.nn as nn
from torch.utils.data import DataLoader

from modelpolaroid.dataset.grid import GridDataset
from modelpolaroid.directioner import Directioner, Direction



cmap_loss = "seismic"
class Polaroid:
    def __init__(
            self, 
            output_folder_plot=None,
            steps=50, 
            max_stepsize=1.5, 
            howmaxstep="boundary",
            origin=None, 
            top_plot=1,
            batch_size=64,
            verbose=True,
            extra_point_to_plot=[],
            device=0):

        assert howmaxstep in ["boundary", "absolute", "adversarial"]

        self.steps = steps
        self.max_stepsize = max_stepsize
        self.howmaxstep = howmaxstep
        self.batch_size = batch_size
        self.top_plot = top_plot
        self.output_folder_plot = output_folder_plot
        self.device = device
        self.verbose = verbose
        self.dict_label_color = {}
        self._list_possible_labels_colors = None
        self.extra_point_to_plot = extra_point_to_plot

        self._softmax = m = nn.Softmax(dim=1)

        if origin is None:
            self.origin = torch.ones((3, 224, 224)).to(device) * 0.5
        else:
            self.origin = origin.to(device)

        self.direction1 = None
        self.direction2 = None

    def reset_color_dict_label(self):
        self.dict_label_color = {}

    def set_output_folder(self, f):
        self.output_folder_plot = f

    def __call__(self, model, direction1="normal", direction2="normal", direction1_kwargs={}, direction2_kwargs={}):
        # Set directions
        print("Set directions.")
        directioner = Directioner(self.origin.shape, bound=(0, 1), origin=self.origin, device=self.device)
        if isinstance(direction1, Direction):
            self.direction1 = direction1
        else:
            self.direction1 = directioner.get_direction(direction1, **direction1_kwargs)

        if isinstance(direction2, Direction):
            self.direction2 = direction2
        else:
            self.direction2 = directioner.get_direction(direction2, former_directions=[self.direction1], **direction2_kwargs)
        
        print("Know when image is out (not in [0, 1]).")
        if self.howmaxstep == "boundary":
            image_distance = 0
            is_out = False
            while not is_out:
                image_distance += 1
                Xf1 = (self.origin + image_distance * self.direction1.direction).flatten(1)
                Xf2 = (self.origin + image_distance * self.direction2.direction).flatten(1)
                is_out = (Xf1.max() >1 or Xf1.min() < 0) and (Xf2.max() >1 or Xf2.min() < 0)
            maxstep = self.max_stepsize * image_distance  
        elif self.howmaxstep == "absolute":
            maxstep = self.max_stepsize
        elif self.howmaxstep == "adversarial":
            p1 = self.direction1.get_perturbation_norm()
            p2 = self.direction2.get_perturbation_norm()
            if p1 is not None and p2 is not None:
                maxstep = self.max_stepsize * max(p1, p2)
            elif p1 is not None:
                maxstep = self.max_stepsize * p1
            elif p2 is not None:
                maxstep = self.max_stepsize * p2
            else:
                raise ValueError("Perturbation norm not defined for at least one direction.")
        
            maxstep = maxstep.cpu().item()
        print("Max step size: {}".format(maxstep))
              
        print("Create dataset.")
        self.dataset = GridDataset(
            self.origin, 
            self.direction1, 
            self.direction2, 
            n_steps=self.steps, 
            max_step=maxstep
            )
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        print("Create Images.")
        losses = []
        labels = []
        diff_loss = []
        X_min = []
        X_max = []
        iterator = tqdm.tqdm(data_loader) if self.verbose else data_loader
        for X in iterator:
            X = X.to(self.device)
            Xf = X.flatten(1)
            X_min.append(Xf.min(1)[0])
            X_max.append(Xf.max(1)[0])
            # is_out = torch.logical_or(Xf.min(1)[0] < 0, Xf.max(1)[0] > 1)

            values = torch.ones((len(X), max(self.top_plot, 2))).to(self.device) * -1
            indices = torch.ones((len(X), max(self.top_plot, 2))).to(self.device).long() * -1

            p = model(X).detach()
            p = self._softmax(p)
            values, indices = torch.topk(p, k=max(self.top_plot, 2), dim=1)
            if self._list_possible_labels_colors is None:
                cm = plt.cm.get_cmap("gist_earth", p.shape[1])
                self._list_possible_labels_colors = [cm(i) for i in range(p.shape[1])]
                random.shuffle(self._list_possible_labels_colors)

            diff_loss.append(values[:, 0] - values[:, 1])
            labels.append(indices[:, :self.top_plot])
            losses.append(values[:, :self.top_plot])

        labels = torch.cat(labels, dim=0).cpu().numpy().reshape((self.steps, self.steps, self.top_plot))
        losses = torch.cat(losses, dim=0).cpu().numpy().reshape((self.steps, self.steps, self.top_plot))
        diff_loss = torch.cat(diff_loss, dim=0).cpu().numpy().reshape((self.steps, self.steps))
        X_min = torch.cat(X_min, dim=0).cpu().numpy().reshape((self.steps, self.steps))
        X_max = torch.cat(X_max, dim=0).cpu().numpy().reshape((self.steps, self.steps))
        X_exist = (X_min >= 0) * (X_max <= 1)
        
        if self.output_folder_plot is not None:
            self._plot_unique(labels, losses, X_exist)
        return labels, losses, X_exist


    def _add_image_points_to_plot(self, ax, add_center=True):
        for dir in [self.direction1, self.direction2]:
            for point, c in dir.get_points():
                x = ((point - self.origin) * self.direction1.direction).sum() / self.direction1.direction.norm()
                y = ((point - self.origin) * self.direction2.direction).sum() / self.direction2.direction.norm()
                x_i = math.floor(x / self.dataset.step_size + self.steps / 2)
                y_i = math.floor(y / self.dataset.step_size + self.steps / 2)
                if x_i >= self.steps or y_i >= self.steps:
                    continue
                
                ax.scatter(x_i, y_i, color="black", s=50, marker="o")

        for point, c in self.extra_point_to_plot:
            point = point.to(self.device)
            x = ((point - self.origin) * self.direction1.direction).sum() / self.direction1.direction.norm()
            y = ((point - self.origin) * self.direction2.direction).sum() / self.direction2.direction.norm()
            x_i = math.floor(x / self.dataset.step_size + self.steps / 2)
            y_i = math.floor(y / self.dataset.step_size + self.steps / 2)
            if x_i >= self.steps or y_i >= self.steps:
                continue
            
            ax.scatter(x_i, y_i, color=c, s=50, marker="1")
        return ax
    

    def _plot_unique(self, labels, losses, X_exist):
        set_labels = set(labels.flatten())
        for l in set_labels:
            if l not in self.dict_label_color:
                self.dict_label_color[l] = self._list_possible_labels_colors[len(self.dict_label_color)]

        colors = ["white"]
        bounds = [-5000]
        for l in sorted(list(self.dict_label_color.keys())):
            colors.append(self.dict_label_color[l])
            bounds.append(l - 0.5)
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
        for i in range(self.top_plot):
            _, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(labels[:, :, i], cmap=cmap, norm=norm)
            ax = self._add_image_points_to_plot(ax)
            plt.axis("off")
            plt.savefig(os.path.join(self.output_folder_plot, "label_{}.png".format(i)))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(losses[:, :, i], vmin=0, vmax=1, cmap=cmap_loss)
            plt.colorbar()
            plt.axis("off")
            plt.savefig(os.path.join(self.output_folder_plot, "loss_{}.png".format(i)))
            plt.close()

            _, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(labels[:, :, i] * X_exist, cmap=cmap, norm=norm)
            ax = self._add_image_points_to_plot(ax)
            plt.axis("off")
            plt.savefig(os.path.join(self.output_folder_plot, "label_{}_real.png".format(i)))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(losses[:, :, i] * X_exist, vmin=0, vmax=1, cmap=cmap_loss)
            plt.colorbar()
            plt.axis("off")
            plt.savefig(os.path.join(self.output_folder_plot, "loss_{}_real.png".format(i)))
            plt.close()
