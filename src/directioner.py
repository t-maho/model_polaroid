import random
import torch

from torchvision import transforms as T

from src.attack.boundary_projection import BP
from src.attack.taig import TAIG
from src.attack.surfree.surfree import SurFree
from src.attack.input_diversity import DI


def get_attack_instance(attack_name="surfree"):
    attack_name = attack_name.strip().lower()
    if attack_name == "surfree":
        return SurFree
    elif attack_name == "bp":
        return BP
    elif attack_name == "taig":
        return TAIG
    elif attack_name == "di":
        return DI
    else:
        raise ValueError
    

class Direction:
    def __init__(self, direction, original=None, adv=None):
        self.original = original
        self.direction = direction
        self.adversarial = adv

        if adv is not None and original is not None:
            self._perturbation_norm = (adv - original).norm()
        else:
            self._perturbation_norm = None
        

    def get_perturbation_norm(self):
        return self._perturbation_norm

    def orthogonalize(self, direction):
        #inner product
        gs_coeff = (self.direction * direction.direction).sum()
        self.direction = self.direction - gs_coeff * direction.direction

    def get_points(self):
        points = []
        if self.adversarial is not None:
            points.append((self.adversarial, "r"))
        if self.original is not None:
            points.append((self.original, "g"))
        return points



class Directioner:
    def __init__(self, shape, bound, origin=None, device="cpu", *args, **kwargs):
        self._origin = origin if origin is not None else torch.zeros(shape).to(device)
        self.device = device
        self._bound = bound
        self._shape = shape
        self.adv = []

    def get_direction(self, dtype="normal", former_directions=[], **kwargs):
        dtype = dtype.lower().strip()
        direction = getattr(self, "get_{}_direction".format(dtype))(**kwargs)    

        for d in former_directions:
            direction.orthogonalize(d)
        return direction


    def get_normal_direction(self, mean=0, std=1, *args, **kwargs):
        x = torch.zeros(self._shape).to(self.device)
        direction = x.normal_(mean, std)
        direction /= direction.norm()
        return Direction(direction)

    def get_sparse_direction(self, *args, **kwargs):
        direction = torch.zeros(self._shape).flatten().to(self.device)
        ind = random.randint(0, len(direction))
        direction[ind] = 1
        direction = direction.reshape(self._shape)
        return Direction(direction)

    def get_image_direction(self, image, *args, **kwargs):
        image = image.to(self.device)
        direction = image - self._origin
        direction /= direction.norm()
        return Direction(direction, original=image)

    def get_attack_direction(self, attack, model, image, *args, **kwargs):
        image = image.unsqueeze(0).to(self.device)
        y = model(image)
        num_classes = y.shape[1]
        attack_inst = get_attack_instance(attack)(device=self.device, num_classes=num_classes, **kwargs)
        adv = attack_inst(model, image, y.argmax(1))
        adv = adv[0]
        image = image[0]
        direction = adv - image
        norm = direction.norm()
        direction /= norm
        print("Attack perturbation norm:", norm.flatten().cpu().numpy())
        print(adv.min())
        print(adv.max())
        print((adv.min() >= 0) * (adv.max() <= 1))
        return Direction(direction, image, adv)

