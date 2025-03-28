from typing import Dict
import torch
import ipdb
import numpy as np
from tqdm import tqdm

from engine.base_tester import BaseTester
from utils import torch_util
from utils.common import get_log_string

class SingleTester(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)
    
    def convert_to_numpy(self, obj):
        """Recursively convert lists back to NumPy arrays."""
        if isinstance(obj, list):
            return [self.convert_to_numpy(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_numpy(value) for key, value in obj.items()}
        return np.array(obj) if isinstance(obj, list) else obj  # Convert lists to arrays
    
    def load_res_dict(self):
        import json
        with open(self.save_path, "r") as json_file:
            res_dict = json.load(json_file)
        return res_dict

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)        
        for iteration, data_dict in pbar:
            # on start
            self.iteration = iteration + 1
            data_dict = torch_util.to_cuda(data_dict)
            self.before_test_step(self.iteration, data_dict)
            # test step
            torch.cuda.synchronize()
            self.timer.add_prepare_time()
            output_dict = self.test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            self.timer.add_process_time()
            # eval step
            results_dict = self.eval_step(self.iteration, data_dict, output_dict)
            message = f'{self.timer.tostring()}'
            pbar.set_description(message)
            torch.cuda.empty_cache()

            self.print_metrics(results_dict)
        
        self.after_test_epoch()

        self.print_metrics(results_dict)

