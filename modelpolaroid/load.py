import requests
import pickle 
import torch
import timm
import os
import torchvision

torch.hub.set_dir("/srv/tempdd/tmaho/models/")


madri_url_dict = {
    "madry_l2=3": {
        "url": "https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0",
        "file": "imagenet_l2_3_0.pt"
    },
    "madry_linf=4": {
        "url": "https://www.dropbox.com/s/axfuary2w1cnyrg/imagenet_linf_4.pt?dl=0",
        "file": "imagenet_linf_4.pt"
    },
    "madry_linf=8": {
        "url": "https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0",
        "file": "imagenet_linf_8.pt"
    }
}


def get_imagenet_labels():
    response = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json")
    return eval(response.content)


def load_data(data_path, device=None):
    data = pickle.load(open(data_path, "rb"))
    print("Images loaded: {}".format(len(data)))
    
    images, labels, starting_points = [], [], []
    for d in data:
        images.append(d["vector"].transpose(2, 0, 1) / 255)
        labels.append(d["label"])
        if "starting_point" in d:
            starting_points.append(d["starting_point"])

    images = torch.Tensor(images)
    labels = torch.Tensor(labels)
    if device is not None and torch.cuda.is_available():
        images = images.cuda(device)
        labels = labels.cuda(device)
    if len(starting_points) == len(images):
        starting_points = torch.Tensor(starting_points)
        if device is not None and torch.cuda.is_available():
            starting_points = starting_points.cuda(device)
    else:
        print("No starting points for all points.")
        starting_points = None

    return images, labels, starting_points



def get_model(model_name, jpeg_module=False, preload_model=None, pretrained=True):
    torch.hub.set_dir("/srv/tempdd/tmaho/torch_models")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
    if preload_model:
        import copy
        normalizer, model = copy.deepcopy(preload_model)
    elif model_name.startswith("madry_"):
        model, std, mean = load_madry(model_name)
    elif model_name.lower().startswith("torchvision"):
        model = getattr(torchvision.models, model_name[len("torchvision_"):])(pretrained=pretrained)
    else:
        model = timm.create_model(model_name, pretrained=pretrained)
        mean = model.default_cfg["mean"]
        std = model.default_cfg["std"]
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
    
    model = torch.nn.Sequential(
                    normalizer,
                    model
                )

    model = model.eval().cuda(0)
    return model


def load_madry(model_name):
    import dill
    # load from https://download.pytorch.org/models/resnet50-19c8e357.pth
    if model_name not in madri_url_dict:
        raise ValueError
    
    weights_path = os.path.join("/nfs/nas4/bbonnet/bbonnet/thibault/extra_model", madri_url_dict[model_name]["file"])

    if not os.path.exists(weights_path):
        r = requests.get(madri_url_dict[model_name]["url"])
        open(weights_path , 'wb').write(r.content)

    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), pickle_module=dill)
    sd = checkpoint["model"]
    for w in ["module.", "attacker.", "model."]:
        sd = {k.replace(w, ""):v for k,v in sd.items()}

    std = sd["normalize.new_std"].flatten()
    mean = sd["normalize.new_mean"].flatten()
    
    del sd["normalize.new_std"]
    del sd["normalize.new_mean"]
    del sd["normalizer.new_std"]
    del sd["normalizer.new_mean"]

    model = torchvision.models.resnet50(pretrained=False)
    model.load_state_dict(sd)
    return model, std, mean

