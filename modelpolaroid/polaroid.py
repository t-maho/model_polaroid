import math
import os
import torch
import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from modelpolaroid.dataset.grid import GridDataset
from modelpolaroid.directioner import Directioner, Direction

class Polaroid:
    def __init__(
            self, 
            output_folder,
            steps=50, 
            max_stepsize=1.5, 
            howmaxstep="boundary",
            origin=None, 
            top_plot=1,
            batch_size=64,
            verbose=True,
            device=0):

        assert howmaxstep in ["boundary", "absolute", "adversarial"]

        self.steps = steps
        self.max_stepsize = max_stepsize
        self.howmaxstep = howmaxstep
        self.batch_size = batch_size
        self.top_plot = top_plot
        self.output_folder = output_folder
        self.device = device
        self.verbose = verbose

        if origin is None:
            self.origin = torch.ones((3, 224, 224)).to(device) * 0.5
        else:
            self.origin = origin.to(device)

        self.direction1 = None
        self.direction2 = None

    def set_output_folder(self, f):
        self.output_folder = f

    def __call__(self, model, direction1="normal", direction2="normal", direction1_kwargs={}, direction2_kwargs={}):
        # Set directions
        print("Set directions.")
        directioner = Directioner(self.origin.shape, bound=(0, 1), origin=self.origin, device=0)
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
            values, indices = torch.topk(p, k=max(self.top_plot, 2), dim=1)

            diff_loss.append(values[:, 0] - values[:, 1])
            labels.append(indices[:, :self.top_plot])
            losses.append(values[:, :self.top_plot])

        labels = torch.cat(labels, dim=0).cpu().numpy().reshape((self.steps, self.steps, self.top_plot))
        losses = torch.cat(losses, dim=0).cpu().numpy().reshape((self.steps, self.steps, self.top_plot))
        diff_loss = torch.cat(diff_loss, dim=0).cpu().numpy().reshape((self.steps, self.steps))
        X_min = torch.cat(X_min, dim=0).cpu().numpy().reshape((self.steps, self.steps))
        X_max = torch.cat(X_max, dim=0).cpu().numpy().reshape((self.steps, self.steps))
        X_exist = (X_min >= 0) * (X_max <= 1)

        self._plot(labels, losses, diff_loss, X_exist)
        self._plot_unique(labels, losses, X_exist)

    def _plot(self, labels, losses, diff_loss, X_exist):

        fig, axs = plt.subplots(self.top_plot, 4, figsize=(6 * self.top_plot, 12))
        for i in range(0, self.top_plot):
            axs[i, 0].imshow(labels[:, :, i], cmap="tab20c")
            axs[i, 0].axis("off")
            axs[i, 0].set_title("Label {}".format(i))

            img_ax = axs[i, 1].imshow(losses[:, :, i])
            axs[i, 1].axis("off")
            axs[i, 1].set_title("Loss {}".format(i))
            fig.colorbar(img_ax, ax=axs[i, 1])


            axs[i, 2].imshow(labels[:, :, i] * X_exist, cmap="tab20c")
            axs[i, 2].axis("off")
            axs[i, 2].set_title("Label {} (Real Image)".format(i))

            img_ax = axs[i, 3].imshow(losses[:, :, i] * X_exist)
            axs[i, 3].axis("off")
            axs[i, 3].set_title("Loss {} (Real Image)".format(i))
            fig.colorbar(img_ax, ax=axs[i, 3])

        axs = self._add_image_points_to_plot(axs)
        plt.savefig(os.path.join(self.output_folder, "labels.png"))
        plt.close()

        fig, axs = plt.subplots(2, figsize=(6, 12))
        img_ax = axs[0].imshow(diff_loss)
        axs[0].axis("off")
        axs[0].set_title("Diff loss")
        fig.colorbar(img_ax, ax=axs[0])

        img_ax = axs[1].imshow(diff_loss * X_exist)
        axs[1].axis("off")
        axs[1].set_title("Diff loss (Real Image)")
        fig.colorbar(img_ax, ax=axs[1])

        plt.savefig(os.path.join(self.output_folder, "diff_loss.png"))
        plt.close()


    def _add_image_points_to_plot(self, axs):
        shape = axs.shape
        for dir in [self.direction1, self.direction2]:
            for point, c in dir.get_points():
                x = ((point - self.origin) * self.direction1.direction).sum() / self.direction1.direction.norm()
                y = ((point - self.origin) * self.direction2.direction).sum() / self.direction2.direction.norm()
                x_i = math.floor(x / self.dataset.step_size + self.steps / 2)
                y_i = math.floor(y / self.dataset.step_size + self.steps / 2)
                if x_i >= self.steps or y_i >= self.steps:
                    continue
                
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        axs[i, j].scatter(x_i, y_i, color=c, s=5 * self.top_plot, marker="1")
        return axs
    

    def _plot_unique(self, labels, losses, X_exist):
        output_unique = os.path.join(self.output_folder, "uniques")
        os.makedirs(output_unique, exist_ok=True)

        for i in range(self.top_plot):
            plt.figure(figsize=(6, 6))
            plt.imshow(labels[:, :, i], cmap="tab20c")
            plt.axis("off")
            plt.savefig(os.path.join(output_unique, "label_{}.png".format(i)))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(losses[:, :, i])
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(output_unique, "loss_{}.png".format(i)))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(labels[:, :, i] * X_exist, cmap="tab20c")
            plt.axis("off")
            plt.savefig(os.path.join(output_unique, "label_{}_real.png".format(i)))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(losses[:, :, i] * X_exist)
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(output_unique, "loss_{}_real.png".format(i)))
            plt.close()
