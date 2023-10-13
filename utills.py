import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib as mpl
import supervision as sv
from typing import Tuple, Union, List
from matplotlib import pyplot as plt

mpl.rcParams['figure.dpi'] = 200  # plot quality
mpl.rcParams['figure.subplot.left'] = 0.01
mpl.rcParams['figure.subplot.right'] = 1 - mpl.rcParams['figure.subplot.left']


def max_show_img_size_reshape(img, max_show_img_size, return_coef=False):  # h,w format
    img_c = np.uint8(img.copy())
    h, w = img_c.shape[:2]
    h_coef = h / max_show_img_size[0]
    w_coef = w / max_show_img_size[1]
    if h_coef < w_coef:  # save the biggest side
        new_img_width = max_show_img_size[1]
        coef = w / new_img_width
        new_img_height = h / coef
    else:
        new_img_height = max_show_img_size[0]
        coef = h / new_img_height
        new_img_width = w / coef
    new_img_height, new_img_width = map(int, [new_img_height, new_img_width])
    img_c = cv2.resize(img_c, (new_img_width, new_img_height), interpolation=cv2.INTER_LINEAR)
    if return_coef:
        return img_c, coef
    return img_c


def plt_show_img(img, title: str = None, add_coef: bool = False, mode: str = 'plt',
                 max_img_size: Tuple[str] = (900, 900)) -> None:
    """
    Display an image using either matplotlib or OpenCV.

    Parameters:
        img (np.ndarray): The image to be displayed.
        title (str, optional): The title of the image. Defaults to None.
        mode (str, optional): The mode to use for displaying the image. It can be either 'plt' for matplotlib or 'cv2' for OpenCV. Defaults to 'plt'.
        max_img_size (Tuple[str], optional): The maximum size of the image to be displayed. Defaults to (900, 900).

    Returns:
        None: This function does not return anything.
    """
    assert mode in ['cv2', 'plt']
    # img_show = img * (255 / img.max()) if add_coef else img.copy()
    img_show = np.interp(img, (img.min(), img.max()), (0, 255)) if add_coef else img.copy()
    img_show = img_show.astype(np.uint8)
    if mode == 'plt':
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    title = str(title) if title is not None else 'image'
    if mode == 'plt':
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(img_show)
        if title:
            ax.set_title(title)
        fig.show()
    elif mode == 'cv2':
        if max_img_size is not None:
            img_show = max_show_img_size_reshape(img_show, max_img_size)
        cv2.imshow(title, img_show)
        cv2.waitKey(0)
        cv2.destroyWindow(title)


def cv2_add_title(img, title, box, color=(0, 255, 0), filled=True, font=cv2.FONT_HERSHEY_COMPLEX, font_scale=0.65,
                  thickness=1, where='top'):
    box = list(map(int, box))
    img = img.copy()
    (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
    if where == 'top':
        text_pos_x, text_pos_y = box[0], box[1] + text_h
    elif where == 'bottom':
        text_pos_x, text_pos_y = box[0], box[3] - 2
    if filled:
        cv2.rectangle(img,
                      (text_pos_x, text_pos_y - text_h - 1),
                      (text_pos_x + text_w, text_pos_y + 4),
                      color, -1)
        cv2.putText(img, title, (text_pos_x, text_pos_y), font, font_scale, (255, 255, 255),
                    thickness)
    else:
        cv2.putText(img, title, (text_pos_x, text_pos_y), font, font_scale, color, thickness)
    return img


def draw(img, item, labels, colors=None, thickness=2, font_scale=1, filled=True, where='bottom'):
    drawed = img.copy()
    categories = list(set(item['objects']['category']))
    if colors is None:
        colors = get_random_colors(len(categories))
    for idx, (bbox, id) in enumerate(zip(item['objects']['bbox'], item['objects']['category'])):
        xmin, ymin, w, h = list(map(int, bbox))
        xmax, ymax = min(xmin + w, item['width']), min(ymin + h, item['height'])
        color = colors[categories.index(id)]

        cv2.rectangle(drawed, (xmin, ymin), (xmax, ymax), color, thickness)
        drawed = cv2_add_title(
            drawed, labels[idx], [xmin, ymin, xmax, ymax], where=where, color=color, filled=filled,
            font=cv2.FONT_HERSHEY_COMPLEX, font_scale=font_scale, thickness=thickness, )
    return drawed


def get_random_colors(n=1):
    colors = []
    for i in range(n):
        randomcolor = (random.randint(0, 150), random.randint(50, 200), random.randint(50, 200))
        colors.append(randomcolor)
    return colors


def glob_search(directories: Union[str, Path, List[str], List[Path]],
                pattern: str = '**/*',
                formats: Union[List[str], Tuple[str], str] = ('png', 'jpg', 'jpeg'),
                shuffle: bool = False,
                seed: int = 2,
                sort: bool = False,
                exception_if_empty=False):
    if isinstance(directories, (str, Path)):
        directories = [Path(directories)]
    files = []
    for directory in directories:
        if isinstance(directory, (str)):
            directory = Path(directory)
        if formats:
            if formats == '*':
                files.extend(directory.glob(f'{pattern}.{formats}'))
            else:
                for format in formats:
                    files.extend(directory.glob(f'{pattern}.{format.lower()}'))
                    files.extend(directory.glob(f'{pattern}.{format.upper()}'))
                    files.extend(directory.glob(f'{pattern}.{format.capitalize()}'))
        else:
            files.extend(directory.glob(f'{pattern}'))
    if exception_if_empty:
        if not len(files):
            raise Exception(f'There are no such files!')
    if shuffle:
        random.Random(seed).shuffle(files)
    if sort:
        files = sorted(files)
    return files
