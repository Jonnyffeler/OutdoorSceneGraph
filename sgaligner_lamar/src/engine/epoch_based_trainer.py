import os
import os.path as osp
from typing import Tuple, Dict
import tqdm
import gc

import ipdb
import torch
import sys


from engine.base_trainer import BaseTrainer
from utils import torch_util
from utils.timer import Timer
from utils.common import get_log_string
from utils.summary_board import SummaryBoard

class EpochBasedTrainer(BaseTrainer):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True, autograd_anomaly_detection=False,
        save_all_snapshots=True, run_grad_check=False, grad_acc_steps=1):
        super().__init__(
            cfg,
            parser=parser,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            save_all_snapshots=save_all_snapshots,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.max_epoch = 1 #cfg.optim.max_epoch
        self.best_val_loss = sys.float_info.max

        self.save_path = "/media/T7/sgaligner/output_self_2/Epoch_Losses_" + "_".join(cfg.modules + cfg.hyperedge_types) + ".txt"
    
    def before_train_step(self, epoch, iteration, data_dict) -> None:
        pass

    def before_val_step(self, epoch, iteration, data_dict) -> None:
        pass

    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def before_train_epoch(self, epoch) -> None:
        pass

    def before_val_epoch(self, epoch) -> None:
        pass

    def after_train_epoch(self, epoch) -> None:
        pass

    def after_val_epoch(self, epoch) -> None:
        pass

    def train_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def val_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def after_backward(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def check_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            self.logger.error('Epoch: {}, iter: {}, invalid gradients.'.format(epoch, iteration))
            torch.save(data_dict, 'data.pth')
            torch.save(self.model, 'model.pth')
            self.logger.error('Data_dict and model snapshot saved.')
            ipdb.set_trace()
    
    def train_epoch(self):
        if self.distributed: 
            self.train_loader.sampler.set_epoch(self.epoch)
        
        self.before_train_epoch(self.epoch)
        self.optimizer.zero_grad()
        total_iterations = len(self.train_loader)
        print(total_iterations)

        
        for iteration, data_dict in enumerate(tqdm.tqdm(self.train_loader)):
            self.inner_iteration = iteration + 1
            self.iteration += 1
            data_dict = torch_util.to_cuda(data_dict)
            self.before_train_step(self.epoch, self.inner_iteration, data_dict)
            self.timer.add_prepare_time()

            # forward
            output_dict, result_dict = self.train_step(self.epoch, self.inner_iteration, data_dict)
            # backward & optimization
            result_dict['loss'].backward(retain_graph=True)
            self.after_backward(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.check_gradients(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.optimizer_step(self.inner_iteration)

            # after training
            self.timer.add_process_time()
            self.after_train_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            self.summary_board.update_from_result_dict(result_dict)

            # logging
            if self.inner_iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = get_log_string(
                    result_dict=summary_dict,
                    epoch=self.epoch,
                    max_epoch=self.max_epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    lr=self.get_lr(),
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event('train', summary_dict, self.iteration)
            
            torch.cuda.empty_cache()
            gc.collect()
            
        self.train_loader.dataset.reset()

        self.after_train_epoch(self.epoch)
        message = get_log_string(self.summary_board.summary(), epoch=self.epoch, timer=self.timer)
        self.logger.critical(message)

        # save all losses 
        with open(self.save_path, "a") as file_write:
            file_write.write(f"{message}\n")
        
        # scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        # snapshot
        self.save_snapshot(f'epoch-{self.epoch}.pth.tar')

    def inference_epoch(self):
        self.set_eval_mode()
        self.before_val_epoch(self.epoch)
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.val_loader)
        pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations)

        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            data_dict = torch_util.to_cuda(data_dict)
            self.before_val_step(self.epoch, self.inner_iteration, data_dict)
            timer.add_prepare_time()
            output_dict, result_dict = self.val_step(self.epoch, self.inner_iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()
            self.after_val_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            
            summary_board.update_from_result_dict(result_dict)
            message = get_log_string(
                result_dict=summary_board.summary(),
                epoch=self.epoch,
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            )
            pbar.set_description(message)
            torch.cuda.empty_cache()
        
        summary_dict = summary_board.summary()
        message = '[Val] ' + get_log_string(summary_dict, epoch=self.epoch, timer=timer)
        val_loss = result_dict['loss']
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_snapshot(f'best_snapshot.pth.tar')

        self.logger.critical(message)
        self.write_event('val', summary_dict, self.epoch)
        self.set_train_mode()

        # save all losses 
        with open(self.save_path, "a") as file_write:
            file_write.write(f"Val: {message}\n")
    
    def set_train_mode(self):
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    def run(self):
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.args.resume:
            self.load_snapshot(osp.join(self.snapshot_dir, 'snapshot.pth.tar'))
        
        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)
        
        self.set_train_mode()
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train_epoch()




        













        


        
    
    