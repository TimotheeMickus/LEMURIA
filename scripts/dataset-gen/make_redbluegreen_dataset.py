# How to use this (disgusting) script:
# For the default browser opened by VPython, make sure you automatically accept
# downloads of PNG files to wherever directory you want the dataset to be in.
# See variable PATH below

import vpython as vp
from time import sleep
from random import uniform

scene = vp.canvas(width=128, height=128)

NUMBER_IMGS = ((3 ** 5) * 1000)

SPHERE = vp.sphere(visible=False)
CUBE = vp.box(visible=False)
RING = vp.ring(visible=False, thickness=0.2)

print('scene set')


def hide_all():
    for obj in [SPHERE, CUBE, RING]:
        obj.visible = False

def random_loc(upness, rightness):
    if upness == 2:
        y = uniform(4.5, 6)
    elif upness == 1:
        y = uniform(-0.75, 0.75)
    else:
        y = uniform(-6, -4.5)
    if rightness == 2:
        x = uniform(4.5, 6)
    elif rightness == 1:
        x = uniform(-0.75, 0.75)
    else:
        x = uniform(-6, -4.5)

    return vp.vector(x,y,uniform(-0.1, 0.1))

def random_orientation():
    x,y,z = uniform(-1., 1.), uniform(-1., 1.), uniform(-1., 1.)
    return vp.vector(x,y,z)

def random_size(bigness):
    if bigness == 2:
        s = uniform(5.0, 6.0)
    elif bigness == 1:
        s = uniform(3., 3.75)
    else:
        s = uniform(1.5, 2)
    return vp.vector(s, s, s)

def random_radius(bigness):
    if bigness == 2:
        s = uniform(4.25, 5.0)
    elif bigness == 1:
        s = uniform(2.5, 3)
    else:
        s = uniform(1.25, 1.75)
    return s


min_margin_dominant_color = 0.7
ceiling_nondominant_color = 0.3
def random_color(blueness):
    if blueness == 2:
        b = uniform(min_margin_dominant_color, 1.)
        g = uniform(0., ceiling_nondominant_color)
        r = uniform(0., ceiling_nondominant_color)
    elif blueness == 1:
        g = uniform(min_margin_dominant_color, 1.)
        b = uniform(0., ceiling_nondominant_color)
        r = uniform(0., ceiling_nondominant_color)
    else:
        r = uniform(min_margin_dominant_color, 1.)
        g = uniform(0., ceiling_nondominant_color)
        b = uniform(0., ceiling_nondominant_color)
    return vp.vector(r, g, b)

def random_obj(obj, upness, rightness, bigness, blueness):
    obj.up = random_orientation()
    obj.pos = random_loc(upness, rightness)
    if obj is RING:
        obj.radius = random_radius(bigness)
    else:
        obj.size = random_size(bigness)
    obj.visible = True
    obj.color = random_color(blueness)
    return obj


_objs = {
    "ring":RING,
    "sphere":SPHERE,
    "cube":CUBE,
}

_colors = {
    "blue":2,
    "green":1,
    "red":0,
}

_up = {
    "up":2,
    "v-mid":1,
    "down":0,
}

_right = {
    "right":2,
    "h-mid":1,
    "left":0,
}

_big = {
    "big":2,
    "medium":1,
    "small":0,
}

def screenshot(fname, buffertime=.125):
    sleep(buffertime)
    scene.capture("%s.png" % fname)
    sleep(buffertime)

scene.camera.pos = vp.vector(0,0,-3)
scene.center = vp.vector(0,0,0)
scene.autoscale=False

import os
import pathlib
# change path below as appropriate
root_dir = pathlib.Path(__file__).absolute().parent.parent.parent
PATH = str(root_dir / 'data/concon/rgb_dataset')
DATASET = {
    os.path.splitext(f)[0]
    for f in os.listdir(PATH)
    if os.path.isfile(os.path.join(PATH, f))
}
print('boot ok')

i = 0
while True:
    for obj in _objs:
        for color in _colors:
            for upness in _up:
                for rightness in _right:
                    for bigness in _big:
                        hide_all()
                        fname = "%i_%s_%s_%s_%s_%s" % (i, obj, color, upness, rightness, bigness)
                        if fname in DATASET:
                            # ignore file previously generated
                            i += 1
                            continue
                        if i >= NUMBER_IMGS:
                            from rm_dups import rm_dups
                            deleted = rm_dups(dataset_path=PATH)
                            if deleted:
                                DATASET = {
                                    os.path.splitext(f)[0]
                                    for f in os.listdir(PATH)
                                    if os.path.isfile(os.path.join(PATH, f))
                                }
                                i = 0
                                print('please relaunch!')
                            exit(0)
                        vp_obj = random_obj(_objs[obj], _up[upness], _right[rightness], _big[bigness], _colors[color])
                        scene.background = vp.color.gray(uniform(0.3, 1.))
                        for l in scene.lights:
                            l.color = vp.color.gray(uniform(0.4, 1.))
                            l.pos = vp.vector(uniform(-5, 5),uniform(-5, 5),uniform(-5, 5))

                        screenshot(fname)
                        i += 1
