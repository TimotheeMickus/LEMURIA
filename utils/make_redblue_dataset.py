import vpython as vp
from time import sleep
from random import uniform

scene = vp.canvas(width=128, height=128)

NUMBER_IMGS = (2 ** 5) * 300

SPHERE = vp.sphere(visible=False)
CUBE = vp.box(visible=False)

def hide_all():
    for obj in [SPHERE, CUBE]:
        obj.visible = False

def random_loc(up, right):
    if up: y = uniform(2.0, 5)
    else: y = uniform(-2.0, -5)
    if right: x = uniform(2.0, 5)
    else: x = uniform(-2.0, -5)

    return vp.vector(x,y,uniform(-0.1, 0.1))

def random_orientation():
    x,y,z = uniform(-1., 1.), uniform(-1., 1.), uniform(-1., 1.)
    return vp.vector(x,y,z)

def random_size(big):
    if big: s = uniform(5.0, 6.0)
    else: s = uniform(1.5, 2)
    return vp.vector(s, s, s)

def random_color(blue):
    if blue:
        b = uniform(.5, 1.)
        g = uniform(0., b - 0.1)
        r = uniform(0., min(b - 0.1, g))
    else:
        r = uniform(.5, 1.)
        g = uniform(0., r - 0.25)
        b = uniform(0., min(r - 0.25, g))
    return vp.vector(r, g, b)



def random_obj(obj, up, right, big, blue):
    obj.up = random_orientation()
    obj.pos = random_loc(up, right)
    obj.size = random_size(big)
    obj.visible = True
    obj.color = random_color(blue)
    return obj


_objs = {
    "sphere":SPHERE,
    "cube":CUBE,
}

_colors = {
    "blue":True,
    "red":False,
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

def screenshot(fname, buffertime=1):
    sleep(buffertime)
    scene.capture("%s.png" % fname)
    sleep(buffertime)

scene.autoscale=False
i = 0
while True:
    for obj in _objs:
        for color in _colors:
            for up in _up:
                for right in _right:
                    for big in _big:
                        hide_all()
                        fname = "%i_%s_%s_%s_%s_%s" % (i, obj, color, up, right, big)
                        vp_obj = random_obj(_objs[obj], _up[up], _right[right], _big[big], _colors[color])
                        scene.background = vp.color.gray(uniform(0.3, 1.))
                        for l in scene.lights:
                            l.color = vp.color.gray(uniform(0.3, 1.))
                            l.pos = vp.vector(uniform(-5, 5),uniform(-5, 5),uniform(-5, 5))
                        screenshot(fname)
                        i += 1
                        if i == NUMBER_IMGS:
                            exit(0)
