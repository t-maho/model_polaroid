import os
import random

from modelpolaroid.load import get_model
from modelpolaroid import Polaroid
from torchvision.io import read_image
from torchvision import transforms as T
import matplotlib.pyplot as plt


output_dir = "./boundary_plot/different_plot_manuscript"
os.makedirs(output_dir, exist_ok=True)

dataset = "train"
random.seed(10)
##################
# Load the model

print("Load model")
model = get_model("efficientnet_b0")

transform = T.Compose([T.Resize(256), T.CenterCrop(224)])

if dataset == "val":
    data_path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/val/"
    img_filename = random.choices(os.listdir(data_path), k=3)
    images = [
        transform(read_image(os.path.join(data_path, img))).float() / 255 
        for img in img_filename]


elif dataset == "train":
    img_dir = "/nfs/nas4/bbonnet/bbonnet/datasets/imagenet_train/"
    images = []
    for i in range(3):
        label = random.choice(os.listdir(img_dir))
        img = random.choice(os.listdir(os.path.join(img_dir, label)))
        images.append(
            transform(read_image(os.path.join(img_dir, label, img))).float() / 255)
        
        plt.imshow(images[-1].permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"image_{i}-label-{label}.png"), bbox_inches="tight")
else:
    raise NotImplementedError

origin = images[0]
directions_param = [
    {
        "name": "attack-normal",
        "origin": images[0],
        "direction1": "attack",
        "direction2": "normal",
        "direction1_kwargs": {"image": images[0], "attack": "bp", "model": model},
        "direction2_kwargs": {"model": model}
    },
    {
        "name": "normal-normal",
        "origin": images[0],
        "direction1": "normal",
        "direction2": "normal",
        "direction1_kwargs": {"model": model},
        "direction2_kwargs": {"model": model}
    },
    {
        "name": "image-image",
        # "origin": images[0],
        "origin": (images[0] + images[1] + images[2]) / 3 ,
        "direction1": "image",
        "direction2": "image",
        "direction1_kwargs": {"image": images[1], "model": model},
        "direction2_kwargs": {"image": images[2], "model": model}
    }
]
steps = 400
top_plot = 2
max_stepsize = 60
for params in directions_param:
    print(params["name"])
    os.makedirs(os.path.join(output_dir, params["name"]), exist_ok=True)


    if params["name"] == "image-image":
        polaroid = Polaroid(
            output_folder=os.path.join(output_dir, params["name"]),
            steps=steps, 
            max_stepsize=max_stepsize, 
            # howmaxstep= "boundary",
            howmaxstep= "absolute",
            origin=params["origin"], 
            top_plot=top_plot,
            batch_size=64, 
            extra_point_to_plot=[(images[0].cuda(0), "g")])
    elif params["name"] == "attack-normal":
        polaroid = Polaroid(
                output_folder=os.path.join(output_dir, params["name"]),
                steps=steps, 
                max_stepsize=max_stepsize, 
                # howmaxstep= "boundary",
                # howmaxstep= "adversarial",
                howmaxstep= "absolute",
                origin=params["origin"], 
                top_plot=top_plot,
                batch_size=64)
    else:
        polaroid = Polaroid(
                output_folder=os.path.join(output_dir, params["name"]),
                steps=steps, 
                max_stepsize=max_stepsize, 
                howmaxstep= "absolute",
                origin=params["origin"], 
                top_plot=top_plot,
                batch_size=64)


    polaroid(
        model, 
        params["direction1"], 
        params["direction2"], 
        direction1_kwargs=params["direction1_kwargs"], 
        direction2_kwargs=params["direction2_kwargs"])