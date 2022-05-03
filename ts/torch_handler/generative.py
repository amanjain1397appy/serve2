"""
Base Handler for Generative Model.
Created by sjw, 2021/7/27.
"""
from PIL import Image
from captum.attr import IntegratedGradients
import base64
import io
import torch
from torchvision import transforms, utils
from ts.torch_handler.vision_handler import VisionHandler


class Generative_Handler(VisionHandler):
    """
    Base class for all generative handlers.
    """
    image_processing = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
