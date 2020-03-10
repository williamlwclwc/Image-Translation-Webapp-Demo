from flask import Flask
import os

app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

from flask import render_template, request, redirect, url_for
from demo.forms import translateHDForm, translateSpadeForm, sampleForm
import random
from shutil import copyfile
import json, base64
from PIL import Image, ImageOps

import torch
from torchvision import transforms
from demo.networks import GlobalGenerator, LocalEnhancer
from demo.networks import SpadeGenerator
generator_global = GlobalGenerator(3, 3)
generator_HD = LocalEnhancer(3, 3, generator_global)
generator_HD.load_state_dict(torch.load("demo/static/model/generatorHD"))
generator_HD.eval()

generator_Spade = SpadeGenerator()
generator_Spade.load_state_dict(torch.load("demo/static/model/generatorSpade"))
generator_Spade.eval()


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")


@app.route("/testset", methods=['get', 'post'])
def testset():
    sample = None

    form_translate_HD = translateHDForm(request.form)
    form_translate_Spade = translateSpadeForm(request.form)
    form_sample = sampleForm(request.form)
    if request.method == 'POST':
        if form_translate_HD.translateHD.data or form_translate_Spade.translateSpade.data:
            pathDir = os.listdir("demo/static/usr_img")
            if len(pathDir) > 0:
                im = Image.open("demo/static/usr_img/seg_map.png")
                im = ImageOps.fit(im, (512, 256), Image.NEAREST)
                im = im.convert("RGB")
                transform_list = [transforms.ToTensor()]
                trans_tensor = transforms.Compose(transform_list)
                im = trans_tensor(im)
                im = im.unsqueeze(0)
                if form_translate_HD.translateHD.data:
                    gen_img = generator_HD(im)
                    print("pix2pixHD")
                else:
                    gen_img = generator_Spade(im)
                    print("Spade")
                trans_PIL = transforms.ToPILImage()
                gen_img = trans_PIL(gen_img[0])
                gen_img.save("demo/static/usr_img/gen_img.jpg", "JPEG")


        if form_sample.sample.data:
            # randomly select an example segmentation map
            pathDir = os.listdir("demo/static/dataset/GT_color/")
            sample = random.sample(pathDir, 1)[0]
            copyfile("demo/static/dataset/GT_color/" + sample, 
                    "demo/static/usr_img/seg_map.png")

    return render_template("testset.html", title="testing dataset", 
                            formTranslateHD=form_translate_HD,
                            formTranslateSpade=form_translate_Spade,
                            formSample=form_sample)


@app.route("/demo", methods=['get', 'post'])
def demo():
    if request.method == "POST":
        recv_data = request.get_json()

        json_re = json.loads(recv_data)
        imgRes = json_re['DrawImg']
        generatorID = json_re['Generator']

        imgdata = base64.b64decode(imgRes)

        with open('demo/static/usr_img/seg_map.png', "wb") as f:
            f.write(imgdata)

        im = Image.open("demo/static/usr_img/seg_map.png")
        im = ImageOps.fit(im, (512, 256), Image.NEAREST)
        im = im.convert("RGB")
        transform_list = [transforms.ToTensor()]
        trans_tensor = transforms.Compose(transform_list)
        im = trans_tensor(im)
        im = im.unsqueeze(0)
        if generatorID == 'pix2pixHD':
            gen_img = generator_HD(im)
            print("pix2pixHD")
        else:
            gen_img = generator_Spade(im)
            print("Spade")
        trans_PIL = transforms.ToPILImage()
        gen_img = trans_PIL(gen_img[0])
        gen_img.save("demo/static/usr_img/gen_img.jpg", "JPEG")

    return render_template("demo.html", title="Demo")

@app.route("/about")
def about():
    return render_template("about.html", title="About")