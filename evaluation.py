# %%
import numpy as np

##### MODEL AND DATA LOADING
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch.utils.data import TensorDataset

import re

import os
import copy

from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str, default=['./saved_models/resnet34/0034_pets'])
parser.add_argument('-model', nargs=1, type=str, default=['200_19push0.8621.pth'])
parser.add_argument('-thr', nargs=1, type=int, default=50)
parser.add_argument('-test_atts', nargs=1, type=str, default=['./data/grad_test_pets.npy']) # python3 main.py -gpuid=0,1,2,3
# args = parser.parse_args(args = """-gpuid 1 
#                                  -modeldir 
#                                  -model """.split())
args = parser.parse_args()                                 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# print(args.modeldir)

# %%
SMOOTH = 1e-6
def iou_array(outputs: np.array, labels: np.array):
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou

def q_mask(att, percentile):
    q_att = []
    for i in range(att.shape[0]):
        tt = att[i]
        q = np.percentile(tt, percentile)
        tt = np.where(tt > q, 0, 1)
        q_att.append(tt)
    q_att = np.stack(q_att, axis=0)
    return q_att

def get_ious_q(atts, model_atts, percentile):
    atts_thr = q_mask(atts, percentile)
    model_thr = q_mask(model_atts, percentile)
    ious = iou_array(atts_thr, model_thr)
    return ious

# %%
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs


class combined_data(torch.utils.data.Dataset):
    def __init__(self, imageset, attentionset, transform=None):
        super(combined_data, self).__init__()
        self.imageset = imageset
        self.attentionset = attentionset
        self.transform = transform
    def __getitem__(self, index):
        concated = torch.cat((self.imageset[index][0], self.attentionset[index][0].unsqueeze(0)), dim=0)
        
        if self.transform != None:
            transformed = self.transform(concated)
        else:
            transformed = concated
        return transformed, self.imageset[index][1]
        # return self.transform(torch.cat((self.imageset[index][0], self.attentionset[index][0].unsqueeze(0)), dim=0)), self.imageset[index][1]
        # return self.imageset[index][0], self.imageset[index][1], self.attentionset[index][0]

    def __len__(self):
        return len(self.attentionset)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# all datasets
# train set


transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), 
        normalize, ])

test_dataset = datasets.ImageFolder(
    test_dir,
    transform_test
    )
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

# print(test_dataset.classes)


test_attentions = TensorDataset(torch.Tensor(np.load(args.test_atts[0])))
test_attn = combined_data(test_dataset, test_attentions)

test_loader = torch.utils.data.DataLoader(
    test_attn, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)


# %%
# print(len(test_dataset))

# %%
# specify the test image to be analyzed
# test_image_dir = args.imgdir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'
# test_image_name = args.img[0] #'Painted_Bunting_0081_15230.jpg'
# test_image_label = args.imgclass[0] #15

# test_image_path = os.path.join(test_image_dir, test_image_name)

# load the model
check_test_accu = False

load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/'
load_model_name = args.model[0] #'10_18push0.7822.pth'

#if load_model_dir[-1] == '/':
#    model_base_architecture = load_model_dir.split('/')[-3]
#    experiment_run = load_model_dir.split('/')[-2]
#else:
#    model_base_architecture = load_model_dir.split('/')[-2]
#    experiment_run = load_model_dir.split('/')[-1]
print(load_model_dir)
model_base_architecture = load_model_dir.split('/')[2]
experiment_run = '/'.join(load_model_dir.split('/')[3:])

# save_analysis_path = os.path.join(test_image_dir, model_base_architecture,
#                                   experiment_run, load_model_name)
save_analysis_path = os.path.join('analysis', model_base_architecture,
                                  experiment_run)
makedir(save_analysis_path)

log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

log('load model from ' + load_model_path)
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run)

ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = ppnet_multi.module.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

# class_specific = False


# %%
start_test = True

all_ious = []

for inputs, labels in test_loader:
    act_patterns = []
    attn = inputs[:, 3, :, :].cuda()
    attn = attn.detach().cpu().numpy()
    inputs = inputs[:, :3, :, :].cuda()
    #inputs = inputs.cuda()
    labels = labels.cuda()


    logits, min_distances, _, _ = ppnet_multi(inputs)
    conv_output, distances = ppnet.push_forward(inputs)
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)

    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    # tables = []
    # for i in range(logits.size(0)):
    #     tables.append((torch.argmax(logits, dim=1)[i].item(), labels[i].item()))
    #     log(str(i) + ' ' + str(tables[-1]))

    for idx in range(logits.size(0)):

        array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
        
        re_act_patterns = []
        for i in range(1,11):
            activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                    interpolation=cv2.INTER_CUBIC)

            # show the image overlayed with prototype activation map
            # rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            # rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)

            re_act_patterns.append(upsampled_activation_pattern)
        
        all_act_patterns = np.stack(re_act_patterns, axis=0)
        polled_act = np.amax(all_act_patterns, axis=0)


        polled_act = polled_act - np.amin(polled_act)
        polled_act = polled_act / np.amax(polled_act)        

        act_patterns.append(polled_act)
    act_patterns = np.array(act_patterns)

    ious = get_ious_q(attn, act_patterns, args.thr)
    if all_ious == []:
        all_ious = ious
    else:
        all_ious = np.append(all_ious, ious)


print('test iou:', np.mean(all_ious))



