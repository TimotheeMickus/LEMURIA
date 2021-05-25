# How to use this (disgusting) script:
# For the default browser opened by VPython, make sure you automatically accept
# downloads of PNG files to wherever directory you want the dataset to be in.
# See variable PATH below

import vpython as vp
from time import sleep
from random import uniform, sample
from itertools import product

scene = vp.canvas(width=128, height=128)

NUMBER_IMGS = ((6 ** 3) + (3 ** 2)) * 300

SPHERE = vp.sphere(visible=False)
CUBE = vp.box(visible=False)
RING = vp.ring(visible=False, thickness=0.2)
HELIX = vp.helix(visible=False, thickness=0.2)
ARROW = vp.arrow(visible=False)
CONE = vp.cone(visible=False)

print('scene set')


def hide_all():
    for obj in [SPHERE, CUBE, RING, HELIX, ARROW, CONE]:
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


COLOR_BOUNDS = [
    [0/3, 1/3 - 1/12], # low
    [1/3 + 1/24, 2/3 - 1/24], # mid
    [2/3 + 1/12, 1.0], # high
]
def random_color(color_desc):
    r, g, b = (
        uniform(*COLOR_BOUNDS[hue_val])
        for hue_val in color_desc
    )
    return vp.vector(r, g, b)

def random_obj(obj, upness, rightness, bigness, object_color):
    obj.up = random_orientation()
    obj.pos = random_loc(upness, rightness)
    if obj is RING:
        obj.radius = random_radius(bigness)
    else:
        obj.size = random_size(bigness)
    obj.visible = True
    obj.color = random_color(object_color)
    return obj


_objs = {
    "ring":RING,
    "sphere":SPHERE,
    "cube":CUBE,
    "helix":HELIX,
    "arrow":ARROW,
    "cone":CONE,
}

_all_colors = sample(list(product([0,1,2], repeat=3)), 12)

_obj_colors = {
    f"obj-col-{i}":c
    for i,c in enumerate(_all_colors[:6])
}

_bg_colors = {
    f"bg-col-{i}":c
    for i,c in enumerate(_all_colors[6:])
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

def screenshot(fname, buffertime=.05):
    sleep(buffertime)
    scene.capture("%s.png" % fname)
    sleep(buffertime)

scene.camera.pos = vp.vector(0,0,-3)
scene.center = vp.vector(0,0,0)
scene.autoscale=False

import os
# change path below as appropriate
PATH = '/home/tmickus/Images/beast'
DATASET = {
    os.path.splitext(f)[0]
    for f in os.listdir(PATH)
    if os.path.isfile(os.path.join(PATH, f))
}
print('boot ok')

i = 0
while True:
    for obj in _objs:
        for obj_color in _obj_colors:
            for bg_color in _bg_colors:
                for upness in _up:
                    for rightness in _right:
                        bigness = "medium"
                        #for bigness in _big:
                        hide_all()
                        fname = f"{i}_{obj_color}_{obj}_{bg_color}_{upness}_{rightness}"
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
                            else:
                                exit(0)
                        vp_obj = random_obj(_objs[obj], _up[upness], _right[rightness], _big[bigness], _obj_colors[obj_color])
                        scene.background = random_color(_bg_colors[bg_color])
                        for l in scene.lights:
                            l.color = vp.color.gray(uniform(0.9, 1.))
                            l.pos = vp.vector(uniform(-5, 5),uniform(-5, 5),uniform(-5, 5))

                        screenshot(fname)
                        i += 1
