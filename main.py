import argparse, os, sys, datetime, glob, importlib, csv
import logging
import numpy as np
import time
import torch
import torchvision
import lightning.pytorch as pl
from src.utils.logger import Logger, setup_logger
from packaging import version
# from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything
from src.utils.cli import CDCLI

from src.data.data_interface import DataModuleFromConfig
from src.detector import CDDector
def cli_main():
    cli = CDCLI(model_class=CDDector,datamodule_class=DataModuleFromConfig)
if __name__ == "__main__":
    cli_main()