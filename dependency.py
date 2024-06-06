import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms