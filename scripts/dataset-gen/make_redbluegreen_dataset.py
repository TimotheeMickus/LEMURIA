import vpython as vp
from time import sleep
from random import uniform

scene = vp.canvas(width=128, height=128)

NUMBER_IMGS = (3 ** 5) * 300

SPHERE = vp.sphere(visible=False)
CUBE = vp.box(visible=False)
HELIX = vp.helix(visible=False)

TERNARY = [0,1,2]

def hide_all():
    for obj in [SPHERE, CUBE, HELIX]:
        obj.visible = False

def random_loc(up, right):
    if up == 2:
        y = uniform(3.0, 5)
    elif up == 1:
        y = uniform(-2.0, 2.0)
    else:
        y = uniform(-5, -3.0)
    if right == 2:
        x = uniform(3.0, 5)
    elif right == 1:
        x = uniform(-2.0, 2)
    else:
        x = uniform(-5, -3.0)

    return vp.vector(x,y,uniform(-0.1, 0.1))

def random_orientation():
    x,y,z = uniform(-1., 1.), uniform(-1., 1.), uniform(-1., 1.)
    return vp.vector(x,y,z)

def random_size(big):
    if big == 2:
        s = uniform(6.0, 7.0)
    elif big == 1:
        s = uniform(3., 4.)
    else:
        s = uniform(1.5, 2)
    return vp.vector(s, s, s)

def random_color(blue):
    if blue == 2:
        b = uniform(.5, 1.)
        g = uniform(0., b - 0.1)
        r = uniform(0., min(b - 0.1, g))
    elif blue == 1:
        g = uniform(.5, 1.)
        b = uniform(0., g - 0.1)
        r = uniform(0., min(g - 0.1, b))
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
    "helix":HELIX,
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
    "mid":1,
    "down":0,
}

_right = {
    "right":2,
    "center":1,
    "left":0,
}

_big = {
    "big":2,
    "medium":1,
    "small":0,
}

def screenshot(fname, buffertime=.5):
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
