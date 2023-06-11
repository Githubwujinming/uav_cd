import logging
import os
import PIL
from typing import Any, Optional
import torch 
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from src.models import hybrid_loss, CDNet
from src.utils.metrics import RunningMetrics
from src.utils.scheduler import CosineWarmupScheduler

class CDDector(pl.LightningModule):
    def __init__(self, detector: nn.Module, nc=2, base_lr=0.01) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = base_lr
        self.detector = detector#CDNet(nc=nc)
        self.training_metrics = RunningMetrics(num_classes=nc)
        self.running_metrics = RunningMetrics(num_classes=nc)

    def forward(self, pre_data, post_data):
        y1, y2, c = self.detector(pre_data, post_data)
        return y1, y2, c


    def training_step(self, batch):
        x1, x2, label, _ = batch
        p1, p2, dist = self(x1, x2)
        bs = x1.shape[0]
        dist = F.interpolate(dist, size=x1.shape[2:], mode='bilinear',align_corners=True)
        pred_L = torch.argmax(dist, dim=1, keepdim=True).long()
        loss = hybrid_loss(dist, label)
        
        self.log('training/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log('training/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, logger=True, batch_size=bs)
        
        self.training_metrics.update(pred_L.detach().cpu().numpy(), label.detach().cpu().numpy())
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        # log metric for sup/unsup data
        sup_score = self.training_metrics.get_scores()

        for k, v in sup_score.items():
            self.log(f'training/{k}', v, on_step=False, on_epoch=True, logger=True)
        
        # logging in file
        train_logger = logging.getLogger('train')
        message = '[Training CD (epoch %d summary)]: F1_1=%.5f \n' %\
                      (self.current_epoch, sup_score['F1_1'])
        for k, v in sup_score.items():
            message += '{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        train_logger.info(message)
        # reset mertric for next epoch
        self.training_metrics.reset()
        
    # 这里配置优化器
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.detector.parameters()),
                                  lr=self.learning_rate)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=1000, max_iters=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x1, x2, label, _ = batch
        p1, p2, dist = self(x1, x2)
        dist = F.interpolate(dist, size=x1.shape[2:], mode='bilinear',align_corners=True)
        pred_L = torch.argmax(dist, dim=1, keepdim=True).long()
        self.running_metrics.update(pred_L.detach().cpu().numpy(), label.detach().cpu().numpy())
       
    @torch.no_grad()
    def on_validation_epoch_end(self):
        # log the validation metrics
        val_logger = logging.getLogger('val')
        scores = self.running_metrics.get_scores()
        self.log('val/F1_1', scores['F1_1'], on_epoch=True,  logger=True)
        self.log('val/iou_1', scores['iou_1'], on_epoch=True, logger=True)
        self.log('val/OA', scores['Overall_Acc'], on_epoch=True,  logger=True)
        self.log('val/precision_1', scores['precision_1'], on_epoch=True, logger=True)
        self.log('val/recall_1', scores['recall_1'], on_epoch=True, logger=True)
        
        message  = f'Validation Summary Epoch: [{self.current_epoch}]\n'
        for k, v in scores.items():
            message += '{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        val_logger.info(message)
        self.running_metrics.reset() 
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x1, x2, label, image_id = batch
        p1, p2, dist = self(x1, x2)
        dist = F.interpolate(dist, size=x1.shape[2:], mode='bilinear',align_corners=True)
        pred = torch.argmax(dist, dim=1, keepdim=True).long()
        
        self.running_metrics.update(pred.detach().cpu().numpy(), label.detach().cpu().numpy())
        # save images
        save_dir = os.path.join(self.trainer.log_dir)
        save_dir = os.path.join(save_dir, 'test_results')
        os.mkdir(save_dir) if not os.path.exists(save_dir) else None
        pred_show = PIL.Image.fromarray(pred.detach().cpu().numpy())
        pred_show.save(os.path.join(save_dir, image_id+'_pred_show.png'))
    
    def on_test_end(self) -> None:
        scores = self.running_metrics.get_scores()
        message = '=========Test: performance=========\n'
        for k, v in scores.items():
            message += '{:s}: {:.4e} '.format(k, v) 
        message += '\n'
        val_logger = logging.getLogger('val')
        val_logger.info(message)
        self.running_metrics.reset()