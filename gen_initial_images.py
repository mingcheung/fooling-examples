import numpy as np
from utils import save_images, load_ground_truth

filenames, _, _ = load_ground_truth("dataset/images.csv")
n = len(filenames)


def gen_randn_images():  # Gaussian distribution
    images = np.zeros((n, 299, 299, 3))
    for i in range(n):
        img = np.random.randn(1, 299, 299, 3)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        images[i] = img
    save_images(images, filenames, "dataset/randn_images")


def gen_rand_images():  # Uniform distribution
    images = np.random.rand(n, 299, 299, 3)
    save_images(images, filenames, "dataset/rand_images")


def gen_white_images():  # White
    images = np.zeros((n, 299, 299, 3)) + 255
    save_images(images, filenames, "dataset/white_images")


def gen_black_images():  # Black
    images = np.zeros((n, 299, 299, 3))
    save_images(images, filenames, "dataset/black_images")


# gen_randn_images()
# gen_rand_images()
gen_white_images()
# gen_black_images()
