"""
Machine Learning Models for 3D Printer Fingerprinting

This module contains functions for initializing various deep learning models,
training them for 3D printer identification, and testing their performance.
Supports multiple architectures including ResNet, EfficientNet, Vision Transformers,
and other modern neural networks.
"""

import copy
import os
import time
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import timm

# Configure model cache directories
torch.hub.set_dir('data/Models')
os.environ["HUGGINGFACE_HUB_CACHE"] = "data/Models"

def set_parameter_requires_grad(model, feature_extracting, freeze_layers):
    """
    Configure which model parameters require gradients during training.
    
    Args:
        model: PyTorch model
        feature_extracting: Whether to extract features (not used in current implementation)
        freeze_layers: If True, freeze all parameters; if False, allow all to be trained
    """
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, freeze_layers=True, classifier_layer_config=0, input_size=224):
    """
    Initialize a deep learning model for 3D printer fingerprinting.
    
    This function supports multiple model architectures and configures them
    for the specific task of printer identification.
    
    Args:
        model_name (str): Name of the model architecture to use
        num_classes (int): Number of printer classes to classify
        feature_extract (bool): Whether to use feature extraction mode
        use_pretrained (bool): Whether to use pretrained weights
        freeze_layers (bool): Whether to freeze backbone layers
        classifier_layer_config (int): Configuration for classifier layers (0, 1, or custom)
        input_size (int): Input image size (default: 224, but 448 is used in practice)
        
    Returns:
        tuple: (model, input_size) - The initialized model and input size
    """
    if model_name == "resnet18":
        model_ft = models.resnet18(weights='DEFAULT')
    if model_name == "resnet34":
        model_ft = models.resnet34(weights='DEFAULT')
    if model_name == "resnet50":
        model_ft = models.resnet50(weights='DEFAULT')
    if model_name == "resnet101":
        model_ft = models.resnet101(weights='DEFAULT')
    if model_name == "resnet152":
        model_ft = models.resnet152(weights='DEFAULT')
    if model_name == "convnext_base":
        model_ft = models.convnext_base(weights='DEFAULT')
    if model_name == "wideresnet50":
        model_ft = models.wide_resnet50_2(weights='DEFAULT')
    if model_name == "efficientnetv2_s":
        model_ft = models.efficientnet_v2_s(weights='DEFAULT')
    if model_name == "efficientnetv2_m":
        model_ft = models.efficientnet_v2_m(weights='DEFAULT')
    if model_name == "swin_v2_t":
        model_ft = models.swin_v2_t(weights='DEFAULT')
    if model_name == "swin_v2_s":
        model_ft = models.swin_v2_s(weights='DEFAULT')
    if model_name == 'convnext_small':
        model_ft = models.convnext_small(weights='DEFAULT')
    if model_name == 'convnext_tiny':
        model_ft = models.convnext_small(weights='DEFAULT')
    if model_name == 'maxvit':
        model_ft = models.maxvit_t(weights='DEFAULT')
    if model_name == 'mobilenet_large':
        model_ft = models.mobilenet_v3_large(weights='DEFAULT')
    if model_name == 'mobilenet_small':
        model_ft = models.mobilenet_v3_small(weights='DEFAULT')
    if model_name == 'vit_b_16':
        model_ft = timm.create_model('vit_base_patch16_384', pretrained=True, img_size=448, num_classes=num_classes)
    if model_name == 'vit_b_32':
        model_ft = models.vit_b_32(weights='DEFAULT')
    if model_name == 'vit_l_16':
        model_ft = models.vit_l_16(weights='DEFAULT')
    if model_name == 'vit_l_32':
        model_ft = models.vit_l_32(weights='DEFAULT')
    if model_name == 'vit_h_14':
        model_ft = models.vit_h_14(weights='DEFAULT')
    if model_name == 'poolformer_m36':
        model_ft = timm.create_model('poolformer_m36.sail_in1k', pretrained=True, num_classes=num_classes)
    if model_name == 'poolformer_m48':
        model_ft = timm.create_model('poolformer_m48', pretrained=True, num_classes=num_classes)
    if model_name == 'poolformer_s36':
        model_ft = timm.create_model('poolformer_s36', pretrained=True, num_classes=num_classes)
    if model_name == 'efficient_vit_m5':
        model_ft = timm.create_model('efficientvit_m5', pretrained=True, num_classes=num_classes)
    if model_name == 'inception_next_base':
        model_ft = timm.create_model('inception_next_base', pretrained=True, num_classes=num_classes)
    if model_name == 'convnextv2_base':
        model_ft = timm.create_model('convnextv2_base.fcmae', pretrained=True, num_classes=num_classes)
    if model_name == 'efficientnetv2_l':
        model_ft = timm.create_model('efficientnetv2_l', num_classes=num_classes)
    if model_name == 'efficientnetv2_xl':
        model_ft = timm.create_model('efficientnetv2_xl', num_classes=num_classes)

    set_parameter_requires_grad(model_ft, feature_extract, freeze_layers)

    if model_name == "convnext_base":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((1024, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "convnext_small":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((768, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "convnext_tiny":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((768, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, 384),
                                            nn.ReLU(),
                                            nn.Linear(384, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "maxvit":
        sequential_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
            nn.Linear(512, 512),
            nn.Tanh(),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(512, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(512, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(512, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)

        model_ft.classifier = sequential_layers
    
    elif model_name == "mobilenet_large":
        sequential_layers = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardshrink(),
            nn.Dropout(p=0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == "mobilenet_small":
        sequential_layers = nn.Sequential(
            nn.Linear(576, 1024, bias=True),
            nn.Hardshrink(),
            nn.Dropout(p=0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)

        model_ft.classifier = sequential_layers

    elif model_name == "efficientnetv2_s":
        sequential_layers = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, 640),
                                            nn.ReLU(),
                                            nn.Linear(640, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == "efficientnetv2_m":
        
        sequential_layers = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            #classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
            classifier_layers=nn.Sequential(nn.Linear(1280, 640),
                                            nn.ReLU(),
                                            nn.Linear(640, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == 'swin_v2_t':
        n_inputs = model_ft.head.in_features
        sequential_layers = nn.Sequential()

        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, int(n_inputs/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(n_inputs/2), num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.head = sequential_layers
    
    elif model_name == 'swin_v2_s':
        n_inputs = model_ft.head.in_features
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.head = sequential_layers

    elif model_name == 'vit_b_16':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, 384),
                                            nn.ReLU(),
                                            nn.Linear(384, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_b_32':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_l_16':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_l_32':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_h_14':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'efficient_vit_m5':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=model_ft.head
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(384, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(384, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'inception_next_base':
        sequential_layers = model_ft.head
        model_ft.head = sequential_layers

    elif model_name == 'convnextv2_base':
        sequential_layers = nn.Sequential(nn.Linear(1024, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, num_classes))
        model_ft.head.fc = sequential_layers

    elif (model_name == 'poolformer_m36') | (model_name == 'poolformer_m48') | (model_name == 'poolformer_s36'):
        sequential_layers = model_ft.head
        model_ft.head = sequential_layers

    else:       
        num_ftrs = model_ft.fc.in_features
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, int(num_ftrs/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(num_ftrs/2), num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.fc = sequential_layers

    input_size = 448

    return model_ft, input_size

def train_model(model, model_name, dataloaders, image_datasets, criterion, optimizer, batch_size, class_names, data_dir, test_samples, device1, scheduler, num_epochs, jigsaw=False, log_interval=1):
    """
    Train a deep learning model for 3D printer fingerprinting.
    
    This function implements the main training loop with validation, model checkpointing,
    and performance tracking. It trains the model to classify 3D printed objects by
    their source printer.
    
    Args:
        model: PyTorch model to train
        model_name (str): Name of the model architecture
        dataloaders (dict): Dictionary containing 'train' and 'val' data loaders
        image_datasets (dict): Dictionary containing image datasets
        criterion: Loss function for training
        optimizer: Optimizer for parameter updates
        batch_size (int): Training batch size
        class_names (list): List of printer class names
        data_dir (str): Directory for saving outputs
        test_samples (int): Number of crops per validation image
        device1: PyTorch device (GPU/CPU)
        scheduler: Learning rate scheduler
        num_epochs (int): Maximum number of training epochs
        jigsaw (bool): Whether to use jigsaw puzzle augmentation (unused)
        log_interval (int): Interval for logging training progress
        
    Returns:
        tuple: (model, val_acc_history, train_loss_history, pred_array_best, 
                label_array_best, best_loss, best_acc)
    """
    model.to(device1)

    since = time.time()
    val_acc_history = []
    train_loss_history = []
    best_acc = 0.0
    best_loss = np.inf
    CELoss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 30)
        running_loss = 0.
        running_corrects = 0.
        model.train()
        phase = 'train'
        print('lr {}'.format(scheduler.get_last_lr()))
        
        # Training loop
        for batch_id, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs, labels = inputs.to(device1), labels.to(device1)

            with torch.autograd.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

            # Log training progress periodically
            if batch_id % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}\tAcc: {}/{}".format(
                    epoch,
                    batch_id * batch_size,
                    len(dataloaders['train'].dataset),
                    100. * batch_id / len(dataloaders['train']),
                    running_loss / ((batch_id + 1) * batch_size),
                    int(running_corrects),
                    batch_id * batch_size
                ))

        scheduler.step()

        # Validation every 10 epochs or on final epoch to save computation time
        # More frequent validation isn't needed as training is stable
        if (epoch % 10 == 0) | (epoch == num_epochs-1):
            val_loss, val_acc, pred_array_final, label_array_final = test_model(
                model, model_name, dataloaders, image_datasets, criterion, epoch, 
                class_names, data_dir, test_samples, device1, epoch_end=True
            )
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            val_acc_history.append(val_acc)
            train_loss_history.append(epoch_loss)

            # Save best model checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                pred_array_best = pred_array_final
                label_array_best = label_array_final

                # Save predictions to CSV file
                df = pd.DataFrame(columns=['img', 'label', 'pred'])
                df.img = (np.array(image_datasets['val'].imgs)[:, 0])
                df.pred = pred_array_best.cpu()
                df.label = label_array_best.cpu()
                print('Outputting csv file')
                pd.DataFrame(df).to_csv(data_dir + "csv_outputs.csv")
                print('Outputting csv file complete')

            print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))
            print()

    # Training complete
    time_elapsed = time.time() - since
    print("Training compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {}".format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_loss_history, pred_array_best, label_array_best, best_loss, best_acc

def voting_sum(array, class_names, device1, filter_vals=None):
    """
    Aggregate predictions across multiple crops using sum voting.
    
    Args:
        array: Tensor of predictions from multiple crops
        class_names: List of class names (unused in current implementation)
        device1: PyTorch device
        filter_vals: Optional filter values (unused)
        
    Returns:
        torch.Tensor: Aggregated predictions
    """
    array = array.to(device1)
    result = torch.sum(array, dim=1)
    max_val, pred = result.max(dim=1)
    return pred


def test_model(model, model_name, dataloaders, image_datasets, criterion, epoch, class_names, data_dir, test_samples, device1, epoch_end=False):
    """
    Evaluate model performance on validation dataset.
    
    This function runs the model in evaluation mode on the validation set,
    handling multiple crops per image and aggregating predictions.
    
    Args:
        model: PyTorch model to evaluate
        model_name (str): Name of the model architecture
        dataloaders (dict): Dictionary containing data loaders
        image_datasets (dict): Dictionary containing image datasets
        criterion: Loss function for evaluation
        epoch (int): Current epoch number
        class_names (list): List of printer class names
        data_dir (str): Directory for saving outputs
        test_samples (int): Number of crops per validation image
        device1: PyTorch device (GPU/CPU)
        epoch_end (bool): Whether this is end-of-epoch evaluation
        
    Returns:
        tuple: (epoch_loss, epoch_acc, pred_part_array, label_part_array)
    """
    model.to(device1)

    running_loss = 0.
    running_corrects = 0.
    model.eval()
    since = time.time()
    pred_array = torch.zeros(0, device=device1)
    label_array = torch.zeros(0, device=device1)
    pred_part_array = torch.zeros(0, device=device1)
    label_part_array = torch.zeros(0, device=device1)
    output_array = torch.zeros(0, device=device1)
    softmax = nn.Softmax(dim=1)
    CELoss = nn.CrossEntropyLoss()

    print('Testing model')

    # Process validation batches
    for batch_id, (inputs, labels) in enumerate(dataloaders['val']):
        inputs, labels = inputs.to(device1), labels.to(device1)
        inputs = inputs.squeeze()
        inputs = inputs.view(-1, 3, 448, 448)

        # Repeat labels for multiple crops per image
        if test_samples > 1:
            labels = (np.repeat(np.array(labels.cpu()), test_samples))
        labels = torch.Tensor(labels).to(device1).long()

        with torch.autograd.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            outputs = softmax(outputs)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * (inputs.size(0) / test_samples)
        running_corrects += int(torch.sum(preds.view(-1) == labels.view(-1)).detach().cpu().numpy()) // test_samples

        # Log progress every 5 batches to monitor progress without overwhelming output
        if batch_id % 5 == 0:
            samples_processed = (batch_id + 1) * (inputs.size(0) // test_samples)
            print("Test Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tPatch Acc: {}/{}".format(
                epoch,
                samples_processed,
                len(dataloaders['val'].dataset),
                100. * samples_processed / len(dataloaders['val'].dataset),
                running_loss / samples_processed,
                int(running_corrects),
                samples_processed
            ))

        # Accumulate predictions and labels
        pred_array = torch.cat((pred_array, preds.view(-1)), 0)
        label_array = torch.cat((label_array, labels.view(-1)), 0)
        output_array = torch.cat((output_array, outputs.view(-1)), 0)

    # Aggregate predictions across crops using voting
    pred_img_array = pred_array.view(-1, test_samples)
    label_img_array = label_array.view(-1, test_samples)
    output_img_array = output_array.view(-1, test_samples, len(class_names))

    # Use sum voting to aggregate predictions
    pred_img_array_mode = voting_sum(output_img_array, class_names, device1).to(device1)
    label_img_array_mode, _ = torch.mode(label_img_array, 1)
    label_img_array_mode = label_img_array_mode.to(device1)

    # Calculate final accuracy
    running_corrects += torch.sum(pred_img_array_mode == label_img_array_mode).detach().cpu().numpy()
    pred_part_array = torch.cat((pred_part_array, pred_img_array_mode), 0)
    label_part_array = torch.cat((label_part_array, label_img_array_mode), 0)

    epoch_loss = running_loss / len(dataloaders['val'].dataset)
    epoch_acc = (running_corrects) / len(dataloaders['val'].dataset)
    
    print('Full Part Accuracy: {}'.format(epoch_acc))

    time_elapsed = time.time() - since
    print("Training compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("{} Loss: {} Acc: {}".format('val', epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc, pred_part_array, label_part_array
