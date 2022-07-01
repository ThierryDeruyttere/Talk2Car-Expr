This the official repository of the Talk2Car-Expr dataset, an **extension** on the Talk2Car dataset.
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
