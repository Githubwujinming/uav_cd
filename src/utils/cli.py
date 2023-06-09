import logging
import os
from typing import Any, Callable, Dict, Optional, Type, Union
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
import pytz
import datetime
import lightning.pytorch as pl
from lightning.pytorch.cli import ArgsType, LightningCLI, LightningArgumentParser, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from src.utils.logger import setup_logger

class CDCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("-n", "--name", type=str, default="", help="name of the experiment")
        parser.add_argument("-r", "--resume", type=str, default="", help="resume from checkpoint")
        parser.add_argument("-p", "--project", type=str, help="name of new or path to existing project")
        parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="experiments",
        help="directory for logging dat shit",
        )
        parser.add_lightning_class_args(ModelCheckpoint, 'CDModelCheckpoint')
        
    def setup_configs(self):
        subcommand = self.subcommand
        name = self.config[subcommand]["name"]
        resume = self.config[subcommand]["resume"]
        logdir = self.config[subcommand]['logdir']
        now = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%m-%d_%H-%M')
        if name and resume:
            raise ValueError(
                "-n/--name and -r/--resume cannot be specified both."
                "If you want to resume training in a new log folder, "
                "use -n/--name in combination with --resume_from_checkpoint"
            )
        if resume:
            if not os.path.exists(resume):
                raise ValueError("Cannot find {}".format(resume))
            if os.path.isfile(resume):
                paths = resume.split("/")
                logdir = "/".join(paths[:-2])
                ckpt = resume
            else:
                assert os.path.isdir(resume), resume
                logdir = resume.rstrip("/")
                # 最新的checkpoint路径
                ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
            # 设置继续训练的checkpoint的路径，后面会保存至trainer_config中，最后用于创建trainer
            _tmp = logdir.split("/")
            nowname = _tmp[-1]
            self.config[subcommand].ckpt_path = ckpt
        else:
            if name:
                name = name + "_"
            else:
                name = ""
            nowname = name + now
            logdir = os.path.join(logdir, nowname)
        # 设置目录
        ckptdir = os.path.join(logdir, "checkpoints")
        #setting logging
        os.makedirs(os.path.join(logdir, 'logs'), exist_ok=True)
        setup_logger('train', os.path.join(logdir, 'logs'), 'train', level=logging.INFO) 
        setup_logger('val', os.path.join(logdir, 'logs'), 'val', level=logging.INFO)
        # 设置模型检查点的保存目录
        self.config[subcommand].CDModelCheckpoint.dirpath = ckptdir
        self.config[subcommand].CDModelCheckpoint.save_last = True
        
        # 设置trainer的log_dir
        self.config[subcommand].trainer.default_root_dir = logdir
        # breakpoint()
    
    def before_instantiate_classes(self) -> None:
        self.setup_configs()
        
    
    def before_fit(self):
        train_logger = logging.getLogger('train')
        train_logger.info(f'================Training started======================')
    
            
    
        