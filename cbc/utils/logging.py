import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

class AverageSummaryWriter:
    def __init__(self, writer=None, log_dir=None, default_period=1, specific_periods={}, prefix=None):
        if(writer is None): writer = SummaryWriter(log_dir)
        else: assert log_dir is None

        self.writer = writer
        self.default_period = default_period
        self.specific_periods = specific_periods
        self.prefix = prefix # If not None, will be added (with ':') before all tags

        self._values = {}

    def reset_values(self):
        self._values = {}

    def add_scalar(self, tag, scalar_value, global_step=None, period=None):
        values = self._values.setdefault(tag, [])
        values.append(scalar_value)

        if(period is None): period = self.specific_periods.get(tag, self.default_period)
        if(len(values) >= period): # If the buffer is full, logs the average and clears the buffer
            _tag = tag if(self.prefix is None) else (self.prefix + ':' +  tag)
            self.writer.add_scalar(tag=_tag, scalar_value=np.mean(values), global_step=global_step)

            values.clear()

    # `l` is a list of pairs (key, value)
    def add_scalar_list(self, l, global_step=None):
        add = False
        for key, value in l:
            self.add_scalar(key, value, global_step)

class Progress:
    def __init__(self, simple_display, steps_per_epoch, epoch, logged_items={"R"}):
        self.simple_display = simple_display
        self.steps_per_epoch = steps_per_epoch
        self.epoch = epoch
        self._logged_items = logged_items

    def __enter__(self):
        if(self.simple_display): self.i = 0
        else: self.pbar = tqdm.tqdm(total=self.steps_per_epoch, postfix={i: 0.0 for i in self._logged_items}, unit="B", desc=("Epoch %i" % self.epoch)) # Do not forget to close it at the end

        return self

    def update(self, num_batches_per_episode, **logged_items):
        if(self.simple_display):
            postfix = " ".join(("%s: %f" % (k, logged_items[k])) for k in sorted(logged_items))
            print(('%i/%i - %s' % (self.i, self.steps_per_epoch, postfix)), flush=True)
            self.i += num_batches_per_episode
        else:
            self.pbar.set_postfix(logged_items, refresh=False)
            self.pbar.update(num_batches_per_episode)

    def __exit__(self, type, value, traceback):
        if(not self.simple_display): self.pbar.close()

class AutoLogger(object):
    def __init__(self, args):
        self.simple_display = args.simple_display
        self.steps_per_epoch = args.steps_per_epoch

        self.current_epoch = 0
        self.global_step = 0
        self.logged_items = {"R"}

        self.summary_writer = AverageSummaryWriter(default_period=args.logging_period)
        self._pbar = None

    def new_progress_bar(self):
        self._pbar = Progress(self.simple_display, self.steps_per_epoch, self.current_epoch, self.logged_items)
        self.current_epoch += 1

    def __enter__(self):
        self.new_progress_bar()
        open(self._pbar)
        return self

    def __exit__(self, type, value, traceback):
        self._pbar.close(type, value, traceback)

    def update(*outcomes, **supplementary_info):
        pass

def log_grads_tensorboard(parameter_list, event_writer):
    """
    Log gradient evolution
    Input:
        `parameter_list`, the model parameters
        `event_writer`, the tensorboard summary
    """
    raw_grad = torch.cat([p.grad.view(-1).detach() for p in parameters])
    median_grad = raw_grad.abs().median().item()
    mean_grad = raw_grad.abs().mean().item()
    max_grad = raw_grad.abs().max().item()

    norm_grad = torch.stack([p.grad.view(-1).detach().data.norm(2.) for p in parameters])
    mean_norm_grad = norm_grad.mean().item()
    max_norm_grad = norm_grad.max().item()
    event_writer.add_scalar('grad/median_grad', median_grad, number_ex_seen)
    event_writer.add_scalar('grad/mean_grad', mean_grad, number_ex_seen)
    event_writer.add_scalar('grad/max_grad', max_grad, number_ex_seen)
    event_writer.add_scalar('grad/mean_norm_grad', mean_norm_grad, number_ex_seen)
    event_writer.add_scalar('grad/max_norm_grad', max_norm_grad, number_ex_seen)
