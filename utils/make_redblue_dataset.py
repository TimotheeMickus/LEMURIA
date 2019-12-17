import vpython as vp
import numpy as np
import time
import random

scene = vp.canvas(width=128, height=128)

NUMBER_IMGS = (2 ** 5) * 300

SPHERE = vp.sphere(visible=False)
CUBE = vp.box(visible=False)

def hide_all():
    for obj in [SPHERE, CUBE]:
        obj.visible = False

def random_loc(up, right):
    if up: y = random.uniform(2.0, 5)
    else: y = random.uniform(-2.0, -5)
    if right: x = random.uniform(2.0, 5)
    else: x = random.uniform(-2.0, -5)

    return vp.vector(x,y,random.uniform(-0.1, 0.1))

def random_orientation():
    x,y,z = random.uniform(-1., 1.), random.uniform(-1., 1.), random.uniform(-1., 1.)
    return vp.vector(x,y,z)

def random_size(obj, big):
    if big:    s = random.uniform(5.0, 6.0)
    else: s = random.uniform(1.5, 2)
    return vp.vector(s, s, s)

def random_obj(obj, up, right, big):
    obj.up = random_orientation()
    obj.pos = random_loc(up, right)
    obj.size = random_size(obj, big)
    obj.visible = True
    return obj


_objs = {
    "sphere":SPHERE,
    "cube":CUBE,
}

_colors = {
    "red":vp.color.red,
    "blue":vp.color.blue,
}

_up = {
    "up":True,
    "down":False,
}

_right = {
    "right":True,
    "left":False,
}

_big = {
    "big":True,
    "small":False,
}


_objs_k = list(_objs)
_colors_k = list(_colors)
_up_k = list(_up)
_right_k = list(_right)
_big_k = list(_big)

def screenshot(fname, buffertime=1):
    time.sleep(buffertime)
    scene.capture("%s.png" % fname)
    time.sleep(buffertime)

scene.autoscale=False
i = 0
while True:
    for obj in _objs:
        for color in _colors:
            for up in _up:
                for right in _right:
                    for big in _big:
                        hide_all()
                        #obj, color, up, right, big = map(random.choice, [_objs_k, _colors_k, _up_k, _right_k, _big_k])
                        fname = "_%i_%s_%s_%s_%s_%s" % (i, obj, color, up, right, big)
                        vp_obj = random_obj(_objs[obj], _up[up], _right[right], _big[big])
                        vp_obj.color = _colors[color]
                        scene.background = vp.color.gray(random.uniform(0.3, 1.))
                        for l in scene.lights:
                            l.color = vp.color.gray(random.uniform(0.3, 1.))
                            l.pos = vp.vector(random.uniform(-5, 5),random.uniform(-5, 5),random.uniform(-5, 5))
                        screenshot(fname)
                        i += 1
                        if i == NUMBER_IMGS:
                            exit(0)
