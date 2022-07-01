from PIL import Image, ImageDraw
import json

img = Image.open("data/train_0.jpg")
drw = ImageDraw.Draw(img)
data = json.load(open("data/talk2car_expr_train.json", "r"))

for command_token, obj_data in data.items():
    if "train_0.jpg" == obj_data["img"]:
        (x,y,w,h) = obj_data["obj_box"]
        drw.rectangle([x,y, x+w, y+h], outline="red")
        img.show()

        print("action: {}".format(obj_data["action"]["name"]))
        print("color: {}".format(obj_data["color"]["name"]))
        print("location: {}".format(obj_data["location"]["name"]))
        print("description: {}".format(obj_data["description"]))
