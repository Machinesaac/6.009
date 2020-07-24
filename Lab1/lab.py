#!/usr/bin/env python3

import math

from PIL import Image


# VARIOUS FILTERS

def get_pixel(image, x, y):
    a = x
    b = y

    if x < 0:
        a = 0
    elif x > image['width'] - 1:
        a = image['width'] - 1

    if y < 0:
        b = 0
    elif y > image['height'] - 1:
        b = image['height'] - 1

    return image['pixels'][a + image['width'] * b]


def set_pixel(image, x, y, c):
    image['pixels'][x + image['width'] * y] = c
    # image['pixels'].append(c)


def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [0] * len(image['pixels']),
    }
    for x in range(image['width']):
        for y in range(image['height']):
            color = get_pixel(image, x, y)
            newcolor = func(color)
            set_pixel(result, x, y, newcolor)

    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255 - c)


# HELPER FUNCTIONS

def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    """
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': ([0] * len(image['pixels'])),
    }
    dis = len(kernel) // 2
    for x in range(image['width']):
        for y in range(image['height']):
            newcolor = 0
            for i in range(len(kernel)):
                for j in range(len(kernel[0])):
                    newcolor += get_pixel(image, x - dis + j, y - dis + i) * kernel[i][j]
            set_pixel(result, x, y, newcolor)

    return result


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    for i in range(len(image['pixels'])):
        image['pixels'][i] = int(round(image['pixels'][i]))
        if image['pixels'][i] < 0:
            image['pixels'][i] = 0
        elif image['pixels'][i] > 255:
            image['pixels'][i] = 255


# FILTERS
def edges(im):
    result = {
        'height': im['height'],
        'width': im['width'],
        'pixels': ([0] * len(im['pixels'])),
    }
    # Initialize two kernels: Kx and Ky
    Kx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    Ky = [[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]
    # computing Ox and Oy by correlating the input with Kx and Ky respectively
    Ox = correlate(im, Kx)
    Oy = correlate(im, Ky)
    # loop over pixels
    for x in range(im['width']):
        for y in range(im['height']):
            # do calculations
            # c is the root of the sum of squares of corresponding pixels in Ox and Oy
            c = ((get_pixel(Ox, x, y)) ** 2 + (get_pixel(Oy, x, y)) ** 2) ** 0.5
            set_pixel(result, x, y, c)
    # ensure that the final image is made up of integer pixels in range [0,255]
    round_and_clip_image(result)
    return result


def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = [[1 / (n * n) for _ in range(n)] for _ in range(n)]

    # then compute the correlation of the input image with that kernel
    result = correlate(image, kernel)

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    round_and_clip_image(result)

    return result


def sharpen(i, n):
    result = {
        'height': i['height'],
        'width': i['width'],
        'pixels': ([0] * len(i['pixels'])),
    }
    kernel = [[1 / (n * n) for _ in range(n)] for _ in range(n)]
    blurred = correlate(i, kernel)
    # iterate over loop and use blurred image do calculation
    for x in range(i['width']):
        for y in range(i['height']):
            # value of sharpened image =
            # 2 * image at location (x,y) - blurred image at location (x,y)
            c = 2 * get_pixel(i, x, y) - get_pixel(blurred, x, y)
            set_pixel(result, x, y, c)
    # ensure that the final image is made up of integer pixels in range [0,255]
    round_and_clip_image(result)
    return result


# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def filt_color(image):
        r, g, b = split(image)
        result = combine(filt(r), filt(g), filt(b))
        return result

    return filt_color


def make_blur_filter(n):
    kernel = [[1 / (n * n) for _ in range(n)] for _ in range(n)]

    def blur(image):
        # then compute the correlation of the input image with that kernel
        result = correlate(image, kernel)
        round_and_clip_image(result)

        return result

    return blur


def make_sharpen_filter(n):
    kernel = [[1 / (n * n) for _ in range(n)] for _ in range(n)]

    def sharp(i):
        result = {
            'height': i['height'],
            'width': i['width'],
            'pixels': ([0] * len(i['pixels'])),
        }
        blurred = correlate(i, kernel)
        # iterate over loop and use blurred image do calculation
        for x in range(i['width']):
            for y in range(i['height']):
                # value of sharpened image =
                # 2 * image at location (x,y) - blurred image at location (x,y)
                c = 2 * get_pixel(i, x, y) - get_pixel(blurred, x, y)
                set_pixel(result, x, y, c)
        # ensure that the final image is made up of integer pixels in range [0,255]
        round_and_clip_image(result)
        return result

    return sharp


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    fs = filters

    def newf(image):
        result = image
        for f in fs:
            result = f(result)

        return result

    return newf


# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """
    im = image
    for i in range(ncols):
        grey = greyscale_image_from_color_image(im)

        energy = compute_energy(grey)

        cem = cumulative_energy_map(energy)

        seam = minimum_energy_seam(cem)

        im = image_without_seam(im, seam)

    return im


# Optional Helper Functions for Seam Carving

def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [],
    }
    for p in image['pixels']:
        v = round(.299 * p[0] + .587 * p[1] + .114 * p[2])
        result['pixels'].append(v)

    return result


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy function),
    computes a "cumulative energy map" as described in the lab 1 writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    image = energy
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': ([0] * len(image['pixels'])),
    }
    for y in range(image['height']):
        for x in range(image['width']):
            if y == 0:
                set_pixel(result, x, y, get_pixel(image, x, y))
            else:
                mini = min(get_pixel(image, x - 1, y - 1),
                           get_pixel(image, x, y - 1),
                           get_pixel(image, x + 1, y - 1))  # get_pixel中已经处理了边界条件

                set_pixel(result, x, y, mini + get_pixel(image, x, y))

    return result


def minimum_energy_seam(c):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 1 writeup).
    """

    image = c
    path = []
    mini = get_pixel(image, 0, image['height'] - 1)
    loc = [0, image['height'] - 1]
    for x in range(image['width']):
        if get_pixel(image, x, image['height'] - 1) < mini:
            mini = get_pixel(image, x, image['height'] - 1)
            loc[0] = x
            loc[1] = image['height'] - 1

    path.append(loc[0] + image['width'] * loc[1])

    for y in range(image['height'] - 1, 0, -1):
        left = get_pixel(image, loc[0] - 1, y - 1)
        mid = get_pixel(image, loc[0], y - 1)
        right = get_pixel(image, loc[0] + 1, y - 1)
        mini = min(left, mid, right)  # get_pixel中已经处理了边界条件
        if mini == left:
            loc[0] = loc[0] - 1
            path.append(loc[0] + image['width'] * (y - 1))
            continue
        if mini == mid:
            path.append(loc[0] + image['width'] * (y - 1))
            continue
        if mini == right:
            loc[0] = loc[0] + 1
            path.append(loc[0] + image['width'] * (y - 1))
            continue

    return path


def image_without_seam(im, s):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    image = im

    result = {
        'height': image['height'],
        'width': image['width'] - 1,
        'pixels': image['pixels'],
    }
    for i in s:
        del result['pixels'][i]

    return result


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES
def split(image):
    r = {
        'height': image['height'],
        'width': image['width'],
        'pixels': ([0] * len(image['pixels']))
    }
    g = {
        'height': image['height'],
        'width': image['width'],
        'pixels': ([0] * len(image['pixels']))
    }
    b = {
        'height': image['height'],
        'width': image['width'],
        'pixels': ([0] * len(image['pixels']))
    }

    for x in range(image['width']):
        for y in range(image['height']):
            set_pixel(r, x, y, get_pixel(image, x, y)[0])
            set_pixel(g, x, y, get_pixel(image, x, y)[1])
            set_pixel(b, x, y, get_pixel(image, x, y)[2])

    return r, g, b


def combine(r, g, b):
    image = {
        'height': r['height'],
        'width': r['width'],
        'pixels': [0] * len(r['pixels'])
    }
    for x in range(image['width']):
        for y in range(image['height']):
            set_pixel(image, x, y, (get_pixel(r, x, y), get_pixel(g, x, y), get_pixel(b, x, y)))
    return image


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    """
    color_inverted = color_filter_from_greyscale_filter(inverted)
    inverted_color_frog = color_inverted(load_color_image('D:/MIT/6.009/Lab1/test_images/frog.png'))
    save_color_image(inverted_color_frog, 'inverted_frog.png')
    """

    """
    blurry = make_blur_filter(9)
    color_blurry = color_filter_from_greyscale_filter(blurry)
    blurry_color_python = color_blurry(load_color_image('D:/MIT/6.009/Lab1/test_images/python.png'))
    save_color_image(blurry_color_python, 'blurry_python.png')
    blurry = make_blur_filter(7)
    color_blurry = color_filter_from_greyscale_filter(blurry)
    blurry_color_python = color_blurry(load_color_image('D:/MIT/6.009/Lab1/test_images/sparrowchick.png'))
    save_color_image(blurry_color_python, 'blurry_sparrowchick.png')
    """

    """
    filter1 = color_filter_from_greyscale_filter(edges)
    filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    filt = filter_cascade([filter1, filter1, filter2, filter1])
    multi_frog = filt(load_color_image('D:/MIT/6.009/Lab1/test_images/frog.png'))
    save_color_image(multi_frog, 'multi_frog.png')
    """

    seam_cats = seam_carving(load_color_image('D:/MIT/6.009/Lab1/test_images/twocats.png'), 100)
    save_color_image(seam_cats, 'seam_cats.png')

    """
    seam_pattern = seam_carving(load_color_image('D:/MIT/6.009/Lab1/test_images/pattern.png'), 1)
    save_color_image(seam_pattern, 'seam_pattern.png')
    """