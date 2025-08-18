
from .domain_bed import DomainBedDataset

import pathlib
import os
import gdown
import tarfile
from zipfile import ZipFile
import json
import shutil


def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


class TerraIncognita(DomainBedDataset):
    name = "terra_incognita"

    url = None
    all_domains = ["38", "43", "46", "100"]
    domain_map = {
        "38": "location_38",
        "43": "location_43",
        "46": "location_46",
        "100": "location_100",
    }

    n_classes = 10

    def download(self):
        # Original URL: https://beerys.github.io/CaltechCameraTraps/
        # New URL: http://lila.science/datasets/caltech-camera-traps

        path = pathlib.Path(self.root) / self.name
        if path.is_dir():
            print("Found existing dataset, skipping download.")
            return

        full_path = stage_path(self.root, self.name)

        download_and_extract(
            "https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz",
            os.path.join(full_path, "terra_incognita_images.tar.gz"))

        download_and_extract(
            "https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip",
            os.path.join(full_path, "caltech_camera_traps.json.zip"))

        include_locations = ["38", "46", "100", "43"]

        include_categories = [
            "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
            "raccoon", "squirrel"
        ]

        images_folder = os.path.join(full_path, "eccv_18_all_images_sm/")
        annotations_file = os.path.join(full_path, "caltech_images_20210113.json")
        destination_folder = full_path

        stats = {}

        if not os.path.exists(destination_folder):
            os.mkdir(destination_folder)

        with open(annotations_file, "r") as f:
            data = json.load(f)

        category_dict = {}
        for item in data['categories']:
            category_dict[item['id']] = item['name']

        for image in data['images']:
            image_location = image['location']

            if image_location not in include_locations:
                continue

            loc_folder = os.path.join(destination_folder,
                                      'location_' + str(image_location) + '/')

            if not os.path.exists(loc_folder):
                os.mkdir(loc_folder)

            image_id = image['id']
            image_fname = image['file_name']

            for annotation in data['annotations']:
                if annotation['image_id'] == image_id:
                    if image_location not in stats:
                        stats[image_location] = {}

                    category = category_dict[annotation['category_id']]

                    if category not in include_categories:
                        continue

                    if category not in stats[image_location]:
                        stats[image_location][category] = 0
                    else:
                        stats[image_location][category] += 1

                    loc_cat_folder = os.path.join(loc_folder, category + '/')

                    if not os.path.exists(loc_cat_folder):
                        os.mkdir(loc_cat_folder)

                    dst_path = os.path.join(loc_cat_folder, image_fname)
                    src_path = os.path.join(images_folder, image_fname)

                    shutil.copyfile(src_path, dst_path)

        shutil.rmtree(images_folder)
        os.remove(annotations_file)

