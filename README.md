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

## Polaroid size

* *output_folder*: folder where images will be saved.
* *steps*: definition of you polaroid. It defines the size of your polaroid which will be of size $steps * steps$
* *max_stepsize*: how far you want to look in the defined directions
* *howmaxstep*: defines the pitch metric:
    * *absolute*: it goes until $steps * direction$
    * *boundary*: it goes until $steps * \lambda * direction$ with $\lambda$, the minimum value for not being an image anymore ( $0 < x < 1$ )
    * *adversarial*: it goes until $steps * p * direction$ with $p$, the perturbation norm to be adversarial in this direction. (**Note: mostly useful when using direction of *attack***)
* *origin*: the origin of you polaroid. It will the center of your polaroid.
* *top_plot*: you can plot the top-k of your model.
* *batch_size*

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