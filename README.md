# Create Images of your model

The package proposes to create a 2d slice of the model.
It will give you a "Polaroid" of your model.

# Setup

Follow these instructions:

```bash
    pip install -r requirements
    python setup.py install
```


# Model Polaroid Package

## Direction

To create the polaroid of your model, two directions have to be chosen.
Four types of directions are currently proposed:
* *normal*: the direction is sampled from the normal distribution. *Parameter: std (default=1), mean (default=0).*
* *sparse*: the direction is only one pixel equal to 1.
* *image*: the direction is given by the image and the origin. *Parameter: image.*
* *attack*: a direction is defined as the normalized perturbation obtained after an adversarial attack. Four attacks are included in the package: SurFree, Boundary Projection, TAIG and DI. *Parameter: model, image*


## Polaroid

Here is an example of a use of the package.

```python
    polaroid = Polaroid(
            output_folder=output_dir,
            steps=200, 
            max_stepsize=1.1, 
            howmaxstep="boundary",
            origin=torch.ones((3, 224, 224)) * 0.5, 
            top_plot=3,
            batch_size=128
    )
    polaroid(
        model=model, 
        direction1="normal", 
        direction2="attack", 
        direction1_kwargs={"mean": 0, "std": 1}, 
        direction2_kwargs={"attack": "bp", "image": image}
        )

```