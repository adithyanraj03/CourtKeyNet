import os
import logging
import datetime
import time
from tensorboardX import SummaryWriter

def setup_logger(save_dir, name):
    """
    Set up logger that prints to console and saves to file
    
    Args:
        save_dir: Directory to save log file
        name: Logger name
    
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'{name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class TensorboardLogger:
    """
    Logger for TensorBoard
    """
    def __init__(self, log_dir):
        """
        Args:
            log_dir: Directory to save tensorboard logs
        """
        self.writer = SummaryWriter(log_dir)
        
    def add_scalar(self, tag, value, step):
        """Add scalar value to tensorboard"""
        self.writer.add_scalar(tag, value, step)
    
    def add_scalars(self, tag_value_dict, step):
        """Add multiple scalar values with same tag"""
        for tag, value in tag_value_dict.items():
            self.writer.add_scalar(tag, value, step)
    
    def add_image(self, tag, img_tensor, step):
        """Add image to tensorboard"""
        self.writer.add_image(tag, img_tensor, step)
    
    def add_figure(self, tag, figure, step):
        """Add matplotlib figure to tensorboard"""
        self.writer.add_figure(tag, figure, step)
    
    def add_histogram(self, tag, values, step):
        """Add histogram to tensorboard"""
        self.writer.add_histogram(tag, values, step)
    
    def add_hparams(self, hparam_dict, metric_dict):
        """Add hyperparameters to tensorboard"""
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def close(self):
        """Close tensorboard writer"""
        self.writer.close()


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        
    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        """Update statistics"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        """String representation"""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """
    Progress meter for tracking batch progress
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        
    def display(self, batch):
        """Display current batch progress"""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def _get_batch_fmtstr(self, num_batches):
        """Get batch format string"""
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'