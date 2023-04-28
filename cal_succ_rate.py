import torch
from torchvision import models
import numpy as np
from utils import Normalize, load_ground_truth

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

filenames, _, labels = load_ground_truth("dataset/images.csv")
# X_adv = np.load("exp_results/single-model-attacks/X_adv_Inc-v3_300iters_Logit_eps32.npy")
X_adv = np.load("exp_results/ensemble-based-attacks/X_adv_Res50_Inc-v3_Dense121_300iters_Logit_white.npy")
X_adv = torch.tensor(X_adv).to(device)
labels = torch.tensor(labels).to(device)

batch_size = 100
num_batches = np.int(np.ceil(len(X_adv) / batch_size))

total_succ = 0
confs_succ = 0.
for i in range(num_batches):
    X_adv_batch = X_adv[i*batch_size: (i+1)*batch_size]
    labels_batch = labels[i*batch_size: (i+1)*batch_size]

    # preds_batch = model_4(norm(X_adv_batch))
    preds_batch = (model_1(norm(X_adv_batch)) + model_2(norm(X_adv_batch)) + model_3(norm(X_adv_batch))) / 3

    confs_batch = torch.softmax(preds_batch, dim=1)
    confs_batch = torch.max(confs_batch, dim=1).values

    idx_succ = torch.argmax(preds_batch, dim=1) == labels_batch

    confs_succ += sum(confs_batch[idx_succ])
    total_succ += sum(idx_succ)

succ_rate = (total_succ / len(labels)).cpu().numpy()
avg_conf = (confs_succ / total_succ).cpu().numpy()
print("      success rate:", succ_rate)
print("average confidence:", avg_conf)
