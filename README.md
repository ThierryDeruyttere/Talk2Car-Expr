This the official repository of the Talk2Car-Expr dataset, an **extension** on the [Talk2Car](https://github.com/talk2car/Talk2Car) dataset, which is built on [nuScenes](https://www.nuscenes.org/).
This dataset adds attributes and referring expressions to the referred objects in Talk2Car.

# Data format
The training and validation data can be found in the `data` folder.

These files are dictionaries with the following data format:

```
{
"command_token":
  {"obj_box": [x, y, w, h],
   "class": "class_name",
   "img": "image.jpg",
   "action": action_value,
   "color": color_value,
   "location": location_value,
   "description": "referring expression"},

   ...
 }
```

The different values for actions, colors and locations can be found in `data/vocabulary.json`.

# How to use

We provide a script, `visualize.py`, to show how to load the data and visualize it.
You can use this dataset as a standalone or together with Talk2Car.
We describe both options below.

## Standalone

If you want to use this dataset as a standalone, please download the images as follows.

First, install gdown.
```
pip install gdown
```

Now download the images

```
gdown --id 1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek
```

Unpack them,

```
unzip imgs.zip && mv imgs/ ./data/images
rm imgs.zip
```

Now the images will be stored in `./data/images`


## With Talk2Car

First, follow the instructions to install Talk2Car as described [here](https://github.com/talk2car/Talk2Car).
Then, copy the train and validation sets found in `./data` to `./data/commands` in Talk2Car.
Now you can load the Talk2Car dataset with Talk2Car_Expr with the following:

```
ds = get_talk2car_class("./data", split="val", load_talk2car_expr=True)
```

This will load the Talk2Car_expr dataset.
To access the data on the commands you can use the following:

```
# Print color of referred object of specific command
print(ds.commands[0].color)

# location
print(ds.commands[0].location)

# action
print(ds.commands[0].action)

```

# Citation

If you use this data, please cite

```
@article{deruyttere2021giving,
  title={Giving commands to a self-driving car: How to deal with uncertain situations?},
  author={Deruyttere, Thierry and Milewski, Victor and Moens, Marie-Francine},
  journal={Engineering Applications of Artificial Intelligence},
  volume={103},
  pages={104257},
  year={2021},
  publisher={Elsevier}
}
```
