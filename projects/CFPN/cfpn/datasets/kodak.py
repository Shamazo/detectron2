import urllib.request
import os
from detectron2.data import DatasetCatalog
import cv2


def download_kodak(root=None):
    """
    Downloads kodak image dataset to root directory
    Args:
        root: directory to download dataset to. If it does not exist defaults to
            the ditrectory in the DETECTRON2_DATASETS environment variable

    Returns:

    """
    if root is None:
        root = os.getenv("DETECTRON2_DATASETS", "./datasets")
    save_dir = os.path.join(root, "kodak")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    count = 0
    for im in range(1, 25):
        save_name = str(im) + ".png"
        save_path = os.path.join(save_dir, save_name)
        if not os.path.exists(save_path):
            if im < 10:
                im_url = "http://r0k.us/graphics/kodak/kodak/kodim0" + str(im) + ".png"
            else:
                im_url = "http://r0k.us/graphics/kodak/kodak/kodim" + str(im) + ".png"
            try:
                print(urllib.request.urlretrieve(im_url, save_path))
                count += 1
            except:
                print("Failed to download: {}".format(im_url))
                pass
        else:
            count += 1
    print("Downloaded  {}/{} images successfully".format(count, 24))


def get_kodak_dicts(img_dir=None):
    """
    Args:
        img_dir: Folder containing the images. Defaults to $DETECTTRON2_DATASETS/kodak

    Returns:

    """
    if img_dir is None:
        root = os.getenv("DETECTRON2_DATASETS", "datasets")
        img_dir = os.path.join(root, "kodak")

    if not os.path.exists(img_dir):
        raise Exception("{} does not exist, please download the dataset \
              manually or use the function download_kodak()".format(img_dir))

    dataset_dicts = []
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        record = {}
        img_path = os.path.join(img_dir, img_name)
        height, width = cv2.imread(img_path).shape[:2]
        record['file_name'] = img_path
        record['width'] = width
        record['height'] = height
        dataset_dicts.append(record)

    return dataset_dicts


def register_kodak(img_dir=None):
    DatasetCatalog.register("kodak_test", lambda: get_kodak_dicts(img_dir=img_dir))
    return
