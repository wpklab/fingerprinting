import torch_optimizer as optim

import os
import pdb
import time
import numpy as np
import gc
import pandas as pd
import typing as t
import random

import matplotlib.pyplot as plt
import torch
#import torch.optim as optim
import torchvision
from torch import nn
from torchvision import datasets, transforms
import ml_models

# from S3N import MultiSmoothLoss

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from PIL import Image

## Testing out Weights and Biases
import wandb
from joblib.externals.loky.backend.context import get_context

import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from matplotlib import pyplot as plt

os.environ["WANDB__SERVICE_WAIT"] = "300"

class random_crop_tensor(torch.nn.Module):
    def __init__(self, scale, scale_2=1):
        self.scale = scale
        self.scale_2 = scale_2
        super(random_crop_tensor, self).__init__()

    def forward(self, img):

        img_w, img_h = img.size
        #Pixel size for scale 1 is 5.29 μm, ROI is 448x448 pixels (2.37 x 2.37 mm)

        img = transforms.ToTensor()(img)
        img_num = 0
        while True:
            img_resize = transforms.Resize((int(img_h/self.scale), int(img_w/self.scale)))(img)

            _, img_resize_w, img_resize_h = img_resize.size()

            img_w_pad = 0
            img_h_pad = 0

            if img_resize_w < 448:
                img_w_pad = int((448 - img_resize_w) / 2)+1

            if img_resize_h < 448:
                img_h_pad = int((448 - img_resize_h) / 2)+1

            padding = max(img_h_pad, img_w_pad)

            img_resize = transforms.Pad(padding)(img_resize)


            img_crop = transforms.RandomCrop(448)(img_resize)

            if self.scale_2 != 1:
                img_crop = transforms.RandomCrop(int(448/self.scale_2))(img_crop)
                img_crop = transforms.Resize(448)(img_crop)


            if (self.scale >= 8) & (self.scale_2 <= 4):
                if (img_crop.mean() > (0.05 / ((img_num // 10) + 1))):
                    return img_crop

            elif (self.scale_2 > 64):
                if (img_crop.mean() > (0.05 / ((img_num // 10) + 1))):
                    return img_crop

            else:
                if (img_crop.mean() > (0.1 / ((img_num // 20) + 1))):
                    return img_crop
            img_num += 1

class random_crop_tensor_test(torch.nn.Module):
    def __init__(self, scale, test_samples, scale_2=1):
        self.scale = scale
        self.test_samples = test_samples
        self.scale_2 = scale_2
        super(random_crop_tensor_test, self).__init__()

    def forward(self, img):     

        x = torch.zeros(self.test_samples, 3, 448, 448)

        img_w, img_h = img.size

        img = transforms.ToTensor()(img)

        
        for idx in range(self.test_samples):

            img_num = 0

            while True:
                img_resize = transforms.Resize((int(img_h/self.scale), int(img_w/self.scale)))(img)

                
                _, img_resize_w, img_resize_h = img_resize.size()

                img_w_pad = 0
                img_h_pad = 0
                
                if img_resize_w < 448:
                    img_w_pad = int((448 - img_resize_w) / 2)+1
                
                if img_resize_h < 448:
                    img_h_pad = int((448 - img_resize_h) / 2)+1

                padding = max(img_h_pad, img_w_pad)

                img_resize = transforms.Pad(padding)(img_resize)


                img_crop = transforms.RandomCrop(448)(img_resize)

                #Optional Recrop
                if self.scale_2 != 1:
                    img_crop = transforms.RandomCrop(int(448/self.scale_2))(img_crop)
                    img_crop = transforms.Resize(448)(img_crop)


                if (self.scale >= 8) & (self.scale_2 <= 4):
                    if (img_crop.mean() > (0.05 / ((img_num // 10) + 1))):
                        x[idx, :, :, :] = torch.Tensor(img_crop)
                        break

                elif (self.scale_2 >= 64):
                    if (img_crop.mean() > (0.05 / ((img_num // 10) + 1))):
                        x[idx, :, :, :] = torch.Tensor(img_crop)
                        break

                else:
                    if (img_crop.mean() > (0.1 / ((img_num // 20) + 1))):
                        x[idx, :, :, :] = torch.Tensor(img_crop)
                        break

                img_num += 1
        return x


def run_model():
    data_dir = 'data'

    gc.collect()
    torch.cuda.empty_cache()

    try:
        wandb.init()
        data_dir = 'data'
        main_dir = 'data'

        device1 = torch.device("cuda") 
        device_id = 0
        #device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        lr = wandb.config.lr #0.0005
        model_name  = wandb.config.model_name
        # wandb.log({'model_name': model_name})
        freeze_layers = False #wandb.config.freeze_layers
        img_size = 448
        weight_decay = wandb.config.weight_decay #0.0001
        scale = wandb.config.scale
        scale_2 = wandb.config.scale_2
        test_samples = wandb.config.test_samples
        lr_gamma = wandb.config.lr_gamma #0.99

        dataset_name = wandb.config.dataset_name
        num_models = 1
        num_workers = 8###Change this based on configuration

        num_epochs = 200
        feature_extract = True

        for c in [dataset_name]:

            ## Mean and STD calculated from training data
            mean = [0.0895, 0.0895, 0.0895]
            std = [0.1101, 0.1101, 0.1101]

            data_transforms = {
                "train": transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    random_crop_tensor(scale=scale, scale_2=scale_2),
                    transforms.Normalize([mean[0], mean[1], mean[2]], [std[0], std[1], std[2]])
                ]),
                "val": transforms.Compose([
                    random_crop_tensor_test(scale=scale, test_samples=test_samples, scale_2=scale_2),
                    transforms.Normalize([mean[0], mean[1], mean[2]], [std[0], std[1], std[2]])
                ]),
            }

            print("Initializing Datasets and Dataloaders...")

            model_path = 'data/Models/'+ str(c) + '_' + wandb.run.id + '.pth'

            ssd_dir = 'data/'+ str(c)

            image_datasets = {x: datasets.ImageFolder(os.path.join(ssd_dir, x), data_transforms[x]) for x in ['train', 'val']}

            class_names = ['4200-1', '5200-1', '5200-2', 'F3B-1', 'F3B-2', 'F3B-3', 'F3B-4', 'F3B-5', 'F3B-6', 'L1-1', 'L1-2', 'M2-1', 'M2-2', 'M2-3', 'M2-4', 'M2-5', 'M2-6', 'S450-1', 'S450-2', 'S900-1', 'S900-2']

            class_names = np.array(class_names)
            print(class_names)

            batch_size = int(16)

            print('Batch Size: ', batch_size)

            wandb.log({'batch_size' : batch_size})

            gc.collect()
            torch.cuda.empty_cache()

            dataloaders_dict = {
                        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, multiprocessing_context=get_context('loky')),
                        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=int(batch_size/test_samples), shuffle=False, num_workers=num_workers, pin_memory=True, multiprocessing_context=get_context('loky')) }

            print(len(image_datasets['train'].imgs))

            num_classes = len(class_names)

            model, input_size = ml_models.initialize_model(model_name, num_classes, feature_extract,
                                                                    use_pretrained=True, freeze_layers=freeze_layers, classifier_layer_config=0)  # initialize_model returns: model_ft, input_size
            #print("Params to learn:")
            if feature_extract:
                params_to_update = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        params_to_update.append(param)


            optimizer_ft = optim.Lamb(params_to_update, lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=lr_gamma)
            criterion = nn.CrossEntropyLoss()

            # Train the model
            model, acc, trainloss, pred_ids, ground_truth_ids, best_val_loss, best_val_acc = ml_models.train_model(model, model_name, dataloaders_dict, image_datasets, criterion, optimizer_ft, batch_size, class_names, main_dir, test_samples, device1, scheduler, num_epochs=num_epochs, jigsaw=jigsaw)
            torch.save(model.state_dict(), model_path)

            # Print results
            results_save_dir = 'data/Results/'+ str(c) + '.txt'
            file_result =  open(results_save_dir, 'w')
            print('loss: ', file=file_result)
            print(trainloss, file=file_result)
            print('acc: ', file=file_result)
            print(acc, file=file_result)
            print('best_acc is: ', file=file_result)
            print(best_val_acc, file=file_result)
            file_result.close()

            # log metrics to wandb
            wandb.log({'max_acc': best_val_acc})
            wandb.log({'min_val_loss': best_val_loss})

            wandb.finish()
    except RuntimeError:
            run_id = wandb.run.id
            wandb.finish()

            api = wandb.Api()
            run = api.run("wpklab/connector_fgc/" + run_id)
            run.delete()

            model = 0

            gc.collect()
            torch.cuda.empty_cache()

import joblib
import wandb

def run_agent():
    wandb.agent(sweep_id="connector_fgc/0drrc7rz", function=run_model)

if __name__ == '__main__':
    num_agents = 2  # Specify the number of agents you want to run

    # run_agent()
    # Use joblib to run multiple agents in parallel
    joblib.Parallel(n_jobs=num_agents)(
        joblib.delayed(run_agent)() for _ in range(num_agents)
    )
    
## Example Sweep Configuration:
# import wandb

# sweep_configuration = {
#     'method': 'grid',
#     'name': 'printer_fingerprinting_21_scale_sample_tests',
#     'metric': {
#         'goal': 'maximize', 
#         'name': 'max_acc'
# 		},
#     'parameters': {
#         'dataset_name': {'values': ['fingerprinting_21_printers']},
#         'model_name': {'values': ['efficientnetv2_m', 'poolformer_m36']},
#         'scale': {'values': [1, 2, 4, 6, 8]},
#         'scale_2': {'values': [1]},
#         'test_samples': {'values': [1, 2, 4, 8, 16]},
#         'lr' : {'values': [0.001, 0.0005]},
#         'lr_gamma': {'values': [0.99]},
#         'weight_decay' : {'values': [0.005]},
#     }  
# }    

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="connector_fgc")