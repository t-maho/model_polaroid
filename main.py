import os
import random

from modelpolaroid.load import get_model
from modelpolaroid import Polaroid
from torchvision.io import read_image
from torchvision import transforms as T


output_dir = "./boundary_plot/"
os.makedirs(output_dir, exist_ok=True)
##################
# Load the model

print("Load model")
model = get_model("efficientnet_b0")

transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
data_path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/val/"
# random.seed(0)
img_filename = random.choices(os.listdir(data_path), k=3)
images = [
    transform(read_image(os.path.join(data_path, img))).float() / 255 
    for img in img_filename]


# origin = None

# direction1 = "attack"
# direction2 = "image"

# direction1_kwargs = {"image": images[0], "attack": "di", "model": model}
# direction2_kwargs = {"image": images[0], "attack": "bp", "model": model}


origin = (images[0] + images[1] + images[2]) / 3
direction1 = "image"
direction2 = "image"
direction1_kwargs = {"image": images[1], "model": model}
direction2_kwargs = {"image": images[2], "model": model}
model = model.cpu()

polaroid = Polaroid(
            output_folder_plot=output_dir,
            steps=50, 
            max_stepsize=1.1, 
            howmaxstep="boundary",
            origin=origin, 
            top_plot=1,
            batch_size=64,
            device="cpu",
            extra_point_to_plot=[(images[0], "black")]
            )

labels, losses, X_exist = polaroid(
    model, 
    direction1, 
    direction2, 
    direction1_kwargs=direction1_kwargs, 
    direction2_kwargs=direction2_kwargs,
    )