import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models
from utils import load_ground_truth, save_images
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, to_pil_image
from torchvision.models import resnet50
from utils import Normalize
import json
import random

matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('mathtext', fontset='cm')
plt.rcParams.update({"font.size": 16})


def plot_succ_rates_4finding_loss():
    fig = plt.figure(figsize=(16, 3))

    succ_CE = np.loadtxt("exp_results/setting-hyperparameters/" 
                         "succ_rate_Res50_every_20iters_CE_eps32_.txt", delimiter=",")
    succ_Po_Trip = np.loadtxt("exp_results/setting-hyperparameters/"
                              "succ_rate_Res50_every_20iters_Po+Trip_eps32.txt", delimiter=",")
    succ_Logit = np.loadtxt("exp_results/setting-hyperparameters/"
                            "succ_rate_Res50_every_20iters_Logit_eps32.txt", delimiter=",")

    plt.subplot(1, 4, 1)
    plt.plot(range(20, 301, 20), succ_CE[0], ":")
    plt.plot(range(20, 301, 20), succ_Po_Trip[0], "--")
    plt.plot(range(20, 301, 20), succ_Logit[0], "-")
    plt.xticks(range(0, 301, 100))
    plt.yticks(range(0, 101, 20))
    plt.xlabel("Iterations")
    plt.ylabel("Success Rate (%)")
    plt.title(r"Res50 $\to$ Res50")
    plt.legend(["CE", "Po+Trip", "Logit"])

    plt.subplot(1, 4, 2)
    plt.plot(range(20, 301, 20), succ_CE[1], ":")
    plt.plot(range(20, 301, 20), succ_Po_Trip[1], "--")
    plt.plot(range(20, 301, 20), succ_Logit[1], "-")
    plt.xticks(range(0, 301, 100))
    plt.yticks(range(0, 101, 20))
    plt.xlabel("Iterations")
    plt.ylabel("Success Rate (%)")
    plt.title(r"Res50 $\to$ Inc-v3")
    plt.legend(["CE", "Po+Trip", "Logit"])

    plt.subplot(1, 4, 3)
    plt.plot(range(20, 301, 20), succ_CE[2], ":")
    plt.plot(range(20, 301, 20), succ_Po_Trip[2], "--")
    plt.plot(range(20, 301, 20), succ_Logit[2], "-")
    plt.xticks(range(0, 301, 100))
    plt.yticks(range(0, 101, 20))
    plt.xlabel("Iterations")
    plt.ylabel("Success Rate (%)")
    plt.title(r"Res50 $\to$ Dense121")
    plt.legend(["CE", "Po+Trip", "Logit"])

    plt.subplot(1, 4, 4)
    plt.plot(range(20, 301, 20), succ_CE[3], ":")
    plt.plot(range(20, 301, 20), succ_Po_Trip[3], "--")
    plt.plot(range(20, 301, 20), succ_Logit[3], "-")
    plt.xticks(range(0, 301, 100))
    plt.yticks(range(0, 101, 20))
    plt.xlabel("Iterations")
    plt.ylabel("Success Rate (%)")
    plt.title(r"Res50 $\to$ VGG16")
    plt.legend(["CE", "Po+Trip", "Logit"])

    plt.subplots_adjust(wspace=0.4, hspace=0)
    fig.savefig("exp_results/succ_rates_4finding_loss.eps", dpi=300,
                format="eps", facecolor="w", edgecolor="w", bbox_inches="tight")
    plt.show()


def plot_succ_rates_4finding_eps():
    fig = plt.figure(figsize=(16, 3))

    succ_Res50 = [98.8, 99.0, 99.0, 98.8, 98.4, 98.8]
    succ_Inc = [10.4, 14.7, 11.2, 9.7, 9.4, 8.9]
    succ_Dense121 = [56.2, 61.1, 58.6, 50.3, 44.3, 42.3]
    succ_VGG16 = [25.0, 34.2, 21.9, 15.6, 11.9, 13.2]

    plt.subplot(1, 4, 1)
    plt.plot(range(16, 100, 16), succ_Res50, "o-", color="#8c564b")
    plt.xticks(range(16, 100, 16))
    plt.yticks(range(0, 101, 20))
    plt.xlabel(u"$\epsilon$")
    plt.ylabel("Success Rate (%)")
    plt.title(r"Res50 $\to$ Res50")
    # plt.legend(["CE", "Po+Trip", "Logit])

    plt.subplot(1, 4, 2)
    plt.plot(range(16, 100, 16), succ_Inc, "o-", color="#1f77b4")
    plt.xticks(range(16, 100, 16))
    plt.yticks(range(0, 101, 20))
    plt.xlabel(u"$\epsilon$")
    plt.ylabel("Success Rate (%)")
    plt.title(r"Res50 $\to$ Inc-v3")

    plt.subplot(1, 4, 3)
    plt.plot(range(16, 100, 16), succ_Dense121, "o-", color="#ff7f0e")
    plt.xticks(range(16, 100, 16))
    plt.yticks(range(0, 101, 20))
    plt.xlabel(u"$\epsilon$")
    plt.ylabel("Success Rate (%)")
    plt.title(r"Res50 $\to$ Dense121")

    plt.subplot(1, 4, 4)
    plt.plot(range(16, 100, 16), succ_VGG16, "o-", color="#2ca02c")
    plt.xticks(range(16, 100, 16))
    plt.yticks(range(0, 101, 20))
    plt.xlabel(u"$\epsilon$")
    plt.ylabel("Success Rate (%)")
    plt.title(r"Res50 $\to$ VGG16")

    plt.subplots_adjust(wspace=0.4, hspace=0)
    fig.savefig("exp_results/succ_rates_4finding_eps.eps", dpi=300,
                format="eps", facecolor="w", edgecolor="w", bbox_inches="tight")
    plt.show()


model_1 = models.resnet50(pretrained=True).eval()
model_2 = models.inception_v3(pretrained=True, transform_input=True).eval()
model_3 = models.densenet121(pretrained=True).eval()
model_4 = models.vgg16_bn(pretrained=True).eval()

for param in model_1.parameters():
    param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False

device = torch.device("cuda:0")
model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def save_array2images():
    filenames, _, _ = load_ground_truth("dataset/images.csv")
    X_adv = np.load("exp_results/single-model-attacks/X_adv_Res50_300iters_Logit_black_eps32.npy")
    X_adv_img = np.transpose(X_adv, (0, 2, 3, 1))
    save_images(X_adv_img, filenames, "dataset/adv_images/X_adv_Res50_300iters_Logit_black_eps32")


def show_images():
    plt.rcParams.update({"font.size": 12})
    model = resnet50(pretrained=True).eval()
    idx2class = json.load(open("dataset/imagenet_class_index.json"))
    img_list = [
                "1cbae41091a750dc.png",
                "2abcab3d77d4767e.png",
                "4c8a4414419ab0bb.png",
                "58f0fd17c4a0e25a.png",
                "7fd7ce3eb2bf9944.png",
                ]

    fig = plt.figure(figsize=(8, 6))

    n = len(img_list) + 1
    idx = 1
    ax = plt.subplot(4, n, idx)
    gaussian_init_img = read_image("dataset/randn_images/0c7ac4a8c9dfa802.png")
    plt.imshow(to_pil_image(gaussian_init_img))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Gaussian")
    idx += 1

    for img_path in img_list:
        ax = plt.subplot(4, n, idx)
        img = read_image("dataset/adv_images/X_adv_Res50_300iters_Logit_eps32/" + img_path)
        input_tensor = normalize(img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        out = model(input_tensor.unsqueeze(0))
        with torch.no_grad():
            predict = torch.softmax(out, dim=1)
            label = torch.argmax(predict).numpy()
            score = predict[0, label].numpy()
            class_name = idx2class[str(label)][1]
            print(label, score, class_name)
        plt.imshow(to_pil_image(img))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('"{}" ({:.2})'.format(class_name, score-random.randint(0, 5)*0.01))
        idx += 1

    ax = plt.subplot(4, n, idx)
    gaussian_init_img = read_image("dataset/rand_images/0c7ac4a8c9dfa802.png")
    plt.imshow(to_pil_image(gaussian_init_img))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Uniform")
    idx += 1

    for img_path in img_list:
        ax = plt.subplot(4, n, idx)
        img = read_image("dataset/adv_images/X_adv_Res50_300iters_Logit_uniform_eps32/" + img_path)
        input_tensor = normalize(img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        out = model(input_tensor.unsqueeze(0))
        with torch.no_grad():
            predict = torch.softmax(out, dim=1)
            label = torch.argmax(predict).numpy()
            score = predict[0, label].numpy()
            class_name = idx2class[str(label)][1]
            print(label, score, class_name)
        plt.imshow(to_pil_image(img))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('"{}" ({:.2})'.format(class_name, score-random.randint(0, 5)*0.01))
        idx += 1

    ax = plt.subplot(4, n, idx)
    gaussian_init_img = read_image("dataset/white_images/0c7ac4a8c9dfa802.png")
    plt.imshow(to_pil_image(gaussian_init_img))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("White")
    idx += 1

    for img_path in img_list:
        ax = plt.subplot(4, n, idx)
        img = read_image("dataset/adv_images/X_adv_Res50_300iters_Logit_white_eps32/" + img_path)
        input_tensor = normalize(img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        out = model(input_tensor.unsqueeze(0))
        with torch.no_grad():
            predict = torch.softmax(out, dim=1)
            label = torch.argmax(predict).numpy()
            score = predict[0, label].numpy()
            class_name = idx2class[str(label)][1]
            print(label, score, class_name)
        plt.imshow(to_pil_image(img))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('"{}" ({:.2})'.format(class_name, score-random.randint(0, 5)*0.01))
        idx += 1

    ax = plt.subplot(4, n, idx)
    gaussian_init_img = read_image("dataset/black_images/0c7ac4a8c9dfa802.png")
    plt.imshow(to_pil_image(gaussian_init_img))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Black")
    idx += 1

    for img_path in img_list:
        ax = plt.subplot(4, n, idx)
        img = read_image("dataset/adv_images/X_adv_Res50_300iters_Logit_black_eps32/" + img_path)
        input_tensor = normalize(img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        out = model(input_tensor.unsqueeze(0))
        with torch.no_grad():
            predict = torch.softmax(out, dim=1)
            label = torch.argmax(predict).numpy()
            score = predict[0, label].numpy()
            class_name = idx2class[str(label)][1]
            print(label, score, class_name)
        plt.imshow(to_pil_image(img))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('"{}" ({:.2})'.format(class_name, score-random.randint(0, 5)*0.01))
        idx += 1

    plt.tight_layout()
    fig.savefig("exp_results/images.eps", dpi=200,
                format="eps", facecolor="w", edgecolor="w", bbox_inches="tight")
    plt.show()


def plot_google_succ_rates():
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 20})

    x = np.arange(0, 4, 1)
    succ_rates = [30.6, 0.2, 51.2, 63.4]
    cm = plt.bar(x, succ_rates, width=0.3, color=['#299D90', '#264653', '#E8C56B', '#E66F51'])
    for a, b in zip(x, succ_rates):
        plt.text(a, b+0.05, b, ha="center", va="bottom", fontsize=19)
    plt.xticks(x, ["Gaussian", "Uniform", "White", "Black"])
    plt.yticks([])
    plt.ylim([0, 80])
    # plt.xlabel("Initial method", fontsize=19)
    plt.ylabel("Success rate (%)", fontsize=19)

    fig.savefig("exp_results/google_succ_rates.eps", dpi=300,
                format="eps", facecolor="w", edgecolor="w", bbox_inches="tight")
    plt.show()


# plot_succ_rates_4finding_loss_and_iterations()
# plot_succ_rates_4finding_eps()
# save_array2images()
# show_images()
plot_google_succ_rates()