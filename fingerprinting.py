"""
Additive Manufacturing Fingerprinting System

This module implements a deep learning system for identifying 3D printer sources
from photographs of printed objects using various neural network architectures.
"""

import os
import gc
import numpy as np
import joblib
import wandb

import torch
import torch_optimizer as optim
from torch import nn
from torchvision import datasets, transforms

from joblib.externals.loky.backend.context import get_context
import ml_models

class random_crop_tensor(torch.nn.Module):
    """
    Custom transform for random cropping and resizing images during training.
    
    This transform applies scaling, padding, and random cropping to images while
    ensuring adequate pixel intensity (to avoid mostly black images).
    
    Args:
        scale (int): Scale factor for image resizing
        scale_2 (int): Additional scale factor for secondary cropping (default: 1)
    """
    
    def __init__(self, scale, scale_2=1):
        super(random_crop_tensor, self).__init__()
        self.scale = scale
        self.scale_2 = scale_2

    def forward(self, img):
        """
        Apply the random crop transformation to an image.
        
        Args:
            img: Input PIL image
            
        Returns:
            torch.Tensor: Transformed image tensor of size (3, 448, 448)
        """
        img_w, img_h = img.size
        # Pixel size for scale 1 is 5.29 μm, ROI is 448x448 pixels (2.37 x 2.37 mm)

        img = transforms.ToTensor()(img)
        img_num = 0
        
        while True:
            # Resize image based on scale factor
            img_resize = transforms.Resize((int(img_h/self.scale), int(img_w/self.scale)), antialias=True)(img)

            _, img_resize_w, img_resize_h = img_resize.size()

            # Calculate padding needed to reach minimum size of 448x448
            img_w_pad = 0
            img_h_pad = 0

            if img_resize_w < 448:
                img_w_pad = int((448 - img_resize_w) / 2) + 1

            if img_resize_h < 448:
                img_h_pad = int((448 - img_resize_h) / 2) + 1

            padding = max(img_h_pad, img_w_pad)
            img_resize = transforms.Pad(padding)(img_resize)

            # Random crop to 448x448
            img_crop = transforms.RandomCrop(448)(img_resize)

            # Apply secondary scaling if specified
            if self.scale_2 != 1:
                img_crop = transforms.RandomCrop(int(448/self.scale_2))(img_crop)
                img_crop = transforms.Resize(448, antialias=True)(img_crop)

            # Check image brightness to avoid mostly black images
            # Different thresholds based on scale factors
            brightness_threshold = 0.05 if (self.scale >= 8 and self.scale_2 <= 4) or self.scale_2 > 64 else 0.1
            adaptive_threshold = brightness_threshold / ((img_num // (10 if self.scale >= 8 and self.scale_2 <= 4 else 20)) + 1)
            
            if img_crop.mean() > adaptive_threshold:
                return img_crop
                
            img_num += 1

class random_crop_tensor_test(torch.nn.Module):
    """
    Custom transform for generating multiple random crops during testing/validation.
    
    This transform generates multiple crops from a single image to increase 
    robustness during evaluation.
    
    Args:
        scale (int): Scale factor for image resizing
        test_samples (int): Number of crops to generate per image
        scale_2 (int): Additional scale factor for secondary cropping (default: 1)
    """
    
    def __init__(self, scale, test_samples, scale_2=1):
        super(random_crop_tensor_test, self).__init__()
        self.scale = scale
        self.test_samples = test_samples
        self.scale_2 = scale_2

    def forward(self, img):
        """
        Generate multiple random crops from an input image.
        
        Args:
            img: Input PIL image
            
        Returns:
            torch.Tensor: Batch of transformed images with shape (test_samples, 3, 448, 448)
        """
        x = torch.zeros(self.test_samples, 3, 448, 448)
        img_w, img_h = img.size
        img = transforms.ToTensor()(img)

        for idx in range(self.test_samples):
            img_num = 0

            while True:
                # Resize image based on scale factor
                img_resize = transforms.Resize((int(img_h/self.scale), int(img_w/self.scale)), antialias=True)(img)
                _, img_resize_w, img_resize_h = img_resize.size()

                # Calculate padding needed to reach minimum size of 448x448
                img_w_pad = 0
                img_h_pad = 0
                
                if img_resize_w < 448:
                    img_w_pad = int((448 - img_resize_w) / 2) + 1
                
                if img_resize_h < 448:
                    img_h_pad = int((448 - img_resize_h) / 2) + 1

                padding = max(img_h_pad, img_w_pad)
                img_resize = transforms.Pad(padding)(img_resize)

                # Random crop to 448x448
                img_crop = transforms.RandomCrop(448)(img_resize)

                # Apply secondary scaling if specified
                if self.scale_2 != 1:
                    img_crop = transforms.RandomCrop(int(448/self.scale_2))(img_crop)
                    img_crop = transforms.Resize(448, antialias=True)(img_crop)

                # Check image brightness to avoid mostly black images
                brightness_threshold = 0.05 if (self.scale >= 8 and self.scale_2 <= 4) or self.scale_2 >= 64 else 0.1
                adaptive_threshold = brightness_threshold / ((img_num // (10 if self.scale >= 8 and self.scale_2 <= 4 else 20)) + 1)
                
                if img_crop.mean() > adaptive_threshold:
                    x[idx, :, :, :] = torch.Tensor(img_crop)
                    break

                img_num += 1
                
        return x


def run_model():
    """
    Main function to train a deep learning model for 3D printer fingerprinting.
    
    This function sets up the dataset, model, and training parameters, then trains
    the model to classify 3D printed objects by their source printer.
    """
    # Clear GPU memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    try:
        # Device configuration
        device1 = torch.device("cuda") 
        device_id = 0
        
        # Hyperparameters
        lr = 0.0005  # Learning rate
        model_name = 'efficientnetv2_m'  # Neural network architecture
        freeze_layers = False  # Whether to freeze pretrained layers
        img_size = 448  # Input image size
        weight_decay = 0.0001  # L2 regularization parameter
        scale = 2  # Image scaling factor for data augmentation
        scale_2 = 1  # Secondary scaling factor
        test_samples = 8  # Number of crops per validation image
        lr_gamma = 0.99  # Learning rate decay factor
        
        main_dir = ''  # Base directory for data

        # Dataset configuration
        dataset_name = 'printer_21_efficiency_10'
        num_models = 1
        num_workers = 8  # Number of data loading workers (adjust based on system)

        num_epochs = 200  # Maximum training epochs
        feature_extract = True  # Use pretrained features

        for c in [dataset_name]:
            # Normalization parameters calculated from training data
            mean = [0.0895, 0.0895, 0.0895]  # RGB channel means
            std = [0.1101, 0.1101, 0.1101]   # RGB channel standard deviations

            # Data transformations for training and validation
            data_transforms = {
                "train": transforms.Compose([
                    transforms.RandomHorizontalFlip(),  # Random horizontal flip augmentation
                    transforms.RandomVerticalFlip(),    # Random vertical flip augmentation
                    random_crop_tensor(scale=scale, scale_2=scale_2),  # Custom crop transform
                    transforms.Normalize([mean[0], mean[1], mean[2]], [std[0], std[1], std[2]])  # Normalization
                ]),
                "val": transforms.Compose([
                    random_crop_tensor_test(scale=scale, test_samples=test_samples, scale_2=scale_2),  # Multiple crops for validation
                    transforms.Normalize([mean[0], mean[1], mean[2]], [std[0], std[1], std[2]])  # Normalization
                ]),
            }

            print("Initializing Datasets and Dataloaders...")

            # Model and data paths
            model_path = str(c) + '.pth'
            ssd_dir = str(c)

            # Create image datasets
            image_datasets = {x: datasets.ImageFolder(os.path.join(ssd_dir, x), data_transforms[x]) for x in ['train', 'val']}

            # 3D printer class names (21 different printers)
            class_names = ['4200-1', '5200-1', '5200-2', 'F3B-1', 'F3B-2', 'F3B-3', 'F3B-4', 'F3B-5', 'F3B-6', 
                          'L1-1', 'L1-2', 'M2-1', 'M2-2', 'M2-3', 'M2-4', 'M2-5', 'M2-6', 'S450-1', 'S450-2', 'S900-1', 'S900-2']
            
            class_names = np.array(class_names)
            print(class_names)

            batch_size = int(16)  # Training batch size
            print('Batch Size: ', batch_size)

            # Clear memory before creating data loaders
            gc.collect()
            torch.cuda.empty_cache()

            # Create data loaders with multiprocessing
            dataloaders_dict = {
                'train': torch.utils.data.DataLoader(
                    image_datasets['train'], 
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=num_workers, 
                    pin_memory=True, 
                    multiprocessing_context=get_context('loky')
                ),
                'val': torch.utils.data.DataLoader(
                    image_datasets['val'], 
                    batch_size=int(batch_size/test_samples),  # Smaller batch size for validation due to multiple crops
                    shuffle=False, 
                    num_workers=num_workers, 
                    pin_memory=True, 
                    multiprocessing_context=get_context('loky')
                )
            }

            print(len(image_datasets['train'].imgs))
            num_classes = len(class_names)

            # Initialize the model
            model, input_size = ml_models.initialize_model(
                model_name, num_classes, feature_extract,
                use_pretrained=True, freeze_layers=freeze_layers, classifier_layer_config=0
            )
            
            # Collect parameters that require gradients for optimization
            if feature_extract:
                params_to_update = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        params_to_update.append(param)

            # Setup optimizer, scheduler, and loss function
            optimizer_ft = optim.Lamb(params_to_update, lr=lr, weight_decay=weight_decay)  # LAMB optimizer
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=lr_gamma)  # Exponential LR decay
            criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification

            # Train the model
            model, acc, trainloss, pred_ids, ground_truth_ids, best_val_loss, best_val_acc = ml_models.train_model(
                model, model_name, dataloaders_dict, image_datasets, criterion, optimizer_ft, 
                batch_size, class_names, main_dir, test_samples, device1, scheduler, 
                num_epochs=num_epochs, jigsaw=False
            )
            
            # Save the trained model
            torch.save(model.state_dict(), model_path)

            # Save training results to text file
            results_save_dir = str(c) + '.txt'
            with open(results_save_dir, 'w') as file_result:
                print('loss: ', file=file_result)
                print(trainloss, file=file_result)
                print('acc: ', file=file_result)
                print(acc, file=file_result)
                print('best_acc is: ', file=file_result)
                print(best_val_acc, file=file_result)

    except RuntimeError as e:
        print(f"Runtime error occurred: {e}")
        model = 0
        # Clear GPU memory on error
        gc.collect()
        torch.cuda.empty_cache()

def run_agent():
    """
    Run a Weights & Biases sweep agent for hyperparameter optimization.
    
    This function connects to a predefined W&B sweep to automatically 
    search for optimal hyperparameters.
    """
    wandb.agent(sweep_id="connector_fgc/0drrc7rz", function=run_model)


if __name__ == '__main__':
    # Number of parallel agents to run (currently set to 1 for single GPU)
    num_agents = 1

    # Run the model training
    run_model()
    
    # Alternative: Run W&B sweep agent (commented out)
    # run_agent()
    
    # Alternative: Run multiple agents in parallel (commented out)
    # joblib.Parallel(n_jobs=num_agents)(
    #     joblib.delayed(run_agent)() for _ in range(num_agents)
    # )


# Example Weights & Biases Sweep Configuration for hyperparameter optimization
# This configuration defines a grid search over various model parameters
"""
import wandb

sweep_configuration = {
    'method': 'grid',
    'name': 'printer_fingerprinting_21_scale_sample_tests',
    'metric': {
        'goal': 'maximize', 
        'name': 'max_acc'
    },
    'parameters': {
        'dataset_name': {'values': ['fingerprinting_21_printers']},
        'model_name': {'values': ['efficientnetv2_m', 'poolformer_m36']},
        'scale': {'values': [1, 2, 4, 6, 8]},
        'scale_2': {'values': [1]},
        'test_samples': {'values': [1, 2, 4, 8, 16]},
        'lr': {'values': [0.001, 0.0005]},
        'lr_gamma': {'values': [0.99]},
        'weight_decay': {'values': [0.005]},
    }  
}    

sweep_id = wandb.sweep(sweep=sweep_configuration, project="connector_fgc")
"""