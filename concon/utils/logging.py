from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import tqdm

from .misc import compute_entropy

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
            self.writer.add_scalar(tag=self.apply_prefix(tag), scalar_value=np.mean(values), global_step=global_step)
            values.clear()

    # `l` is a list of pairs (key, value)
    def add_scalar_list(self, l, global_step=None):
        add = False
        for key, value in l:
            self.add_scalar(key, value, global_step)

    def apply_prefix(self, tag):
        return tag if(self.prefix is None) else ('%s-%s' % (self.prefix, tag))

# TM: for now, I'm using the `display` attribute in the code.
class Progress(metaclass=ABCMeta):
    def __init__(self, steps_per_epoch, epoch, logged_items={"R"}):
        self.steps_per_epoch = steps_per_epoch
        self.epoch = epoch
        self._logged_items = logged_items

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass

    @abstractmethod
    def update(self, **logged_items):
        pass


    # yucky reverse-pattern monkey-patch for now
    @staticmethod
    def get_progress_cls(display):
        """
        Retrieves correct class based on display
        """
        progress_cls = None
        if display=="tqdm": progress_cls = TQDMProgress
        elif display=="simple": progress_cls = SimpleProgress # used to be "normal"
        # TODO: how should minimal be handled?
        elif display=="minimal": progress_cls = SimpleProgress
        assert progress_cls is not None, "the `display` parameter is invalid"
        return progress_cls


class TQDMProgress(Progress):
    def __init__(self, steps_per_epoch, epoch, logged_items={"R"}):
        super().__init__(steps_per_epoch, epoch, logged_items={"R"})

    def __enter__(self):
        self.pbar = tqdm.tqdm(
            total=self.steps_per_epoch,
            postfix={i: 0.0 for i in self._logged_items},
            unit="B",
            desc=("Epoch %i" % self.epoch)) # Do not forget to close it at the end
        return self

    def update(self, **logged_items):
        self.pbar.set_postfix(logged_items, refresh=False)
        self.pbar.update()

    def __exit__(self, type, value, traceback):
        self.pbar.close()

class SimpleProgress(Progress):
    def __init__(self, steps_per_epoch, epoch, logged_items={"R"}):
        super().__init__(steps_per_epoch, epoch, logged_items={"R"})

    def __enter__(self):
        self.i = 0
        return self

    def update(self, **logged_items):
        postfix = " ".join(("%s: %f" % (k, logged_items[k])) for k in sorted(logged_items))
        print(('%i/%i - %s' % (self.i, self.steps_per_epoch, postfix)), flush=True)
        self.i += 1

class DummyLogger(object):
    """
        Place holder class
    """

    def __enter__(self):
        pass

    def __exit__(self, *vargs, **kwargs):
        pass

    def update(self, *vargs, **kwargs):
        pass

    def _write(self, *vargs, **kwargs):
        pass

# TODO Could have a 'print(self, message, flush, type)' method
class AutoLogger(object):
    def __init__(self, base_alphabet_size, data_loader, display='tqdm', steps_per_epoch=1000, log_debug=False, log_lang_progress=False, log_entropy=False, device='cpu', no_summary=False, summary_dir=None, default_period=10,):

        self.base_alphabet_size = base_alphabet_size
        self.data_loader = data_loader

        self.display = display
        self.steps_per_epoch = steps_per_epoch

        self.current_epoch = 0
        self.global_step = 0
        self.logged_items = {"S"}

        self.log_debug = log_debug
        self.log_lang_progress = log_lang_progress
        self.log_entropy = log_entropy

        self.device = device

        if(no_summary):
            self.summary_writer = None
        else:
            self.summary_writer = AverageSummaryWriter(log_dir=summary_dir, default_period=default_period)
        self._pbar = None

        self._state = {}

        self.tag_header = ""

    def new_progress_bar(self):
        """
        Initializes a progress bar.
        """
        progress_cls = Progress.get_progress_cls(self.display)
        self._pbar = progress_cls(self.steps_per_epoch, self.current_epoch, self.logged_items)
        self.current_epoch += 1

    def __enter__(self):
        self.new_progress_bar()
        self._state = {
            'total_reward' : 0.0, # sum of the rewards since the beginning of the epoch
            'total_success' : 0.0, # sum of the successes since the beginning of the epoch
            'total_items' : 0, # number of training instances since the beginning of the epoch
            'running_avg_reward' : 0.0,
            'running_avg_success' : 0.0,
        }
        if self.summary_writer is not None:
            if self.log_lang_progress:
                self._state['current_dist'] = torch.zeros((self.base_alphabet_size, 5), dtype=torch.float).to(self.device)
                self._state['past_dist'] = None # size of embeddings past_dist, current_dist = None,

            if self.log_entropy:
                self._state['symbol_counts'] = torch.zeros(self.base_alphabet_size, dtype=torch.float).to(self.device)

        self._pbar.__enter__()

        return self

    def _write(self, tag, val, step, direct=False):
        """Convenience method for writing to summary. Prepends with tag_header on the fly if necessary.
            args:
                tag, val, step: for summary
            direct: target summary_writer.writer (wrapped object) instead of summary_writer
        """
        if self.summary_writer is not None:
            writer = self.summary_writer.writer if direct else self.summary_writer
            if(direct): tag = self.summary_writer.apply_prefix(tag)
            return writer.add_scalar(tag, val, step)

    def __exit__(self, type, value, traceback):
        if self.log_entropy and self.summary_writer is not None:
            self._write('llp/language_entropy', compute_entropy(self._state['symbol_counts'], base=2), self._state['number_ex_seen'], direct=True)

        self._pbar.__exit__(type, value, traceback)
        self._state = {}


    def update(self, loss, *external_output, **supplementary_info):
        # TODO: the autologger needs some clean up for user simplicity. Ideally I'd love to have it in its own thread as well

        rewards, successes, msg_length, sender_entropy, receiver_entropy, *external_output = external_output

        # Computes the minimum length the messages can have in order to get perfect accuracy (approximation when the size of the alphabet >> 1)
        minimal_compression_len = np.log(self.data_loader.nb_categories) / np.log(self.base_alphabet_size + 1) # + 1 because EoM is taken into account
        length_ratio = (msg_length / minimal_compression_len).item()

        # updates running average reward
        self._state['total_reward'] += rewards.sum().item()
        self._state['total_success'] += successes.sum().item()
        self._state['total_items'] += supplementary_info['batch'].size
        self._state['running_avg_reward'] = self._state['total_reward'] / self._state['total_items']
        self._state['running_avg_success'] = self._state['total_success'] / self._state['total_items']

        if self.summary_writer is not None:
            avg_reward = rewards.mean().item() # average reward of the batch
            avg_success = successes.mean().item() # average success of the batch
            number_ex_seen = supplementary_info['index'] * supplementary_info['batch'].size
            self._state['number_ex_seen'] = number_ex_seen

            self._write('train/reward', avg_reward, number_ex_seen)
            self._write('train/success', avg_success, number_ex_seen)
            self._write('train/loss', loss.item(), number_ex_seen)
            self._write('train/sender_entropy', sender_entropy.item(), number_ex_seen)
            self._write('train/receiver_entropy', receiver_entropy.item(), number_ex_seen)

            self._write('llp/msg_length', msg_length.item(), number_ex_seen)
            self._write('llp/length_ratio', length_ratio, number_ex_seen)

            if self.log_lang_progress:
                for batch in supplementary_info['batches']:
                    batch_msg_manyhot = torch.zeros((batch.size, self.base_alphabet_size + 2), dtype=torch.float).to(self.device) # size of embeddings + EOS + PAD
                    # message -> many-hot
                    many_hots = batch_msg_manyhot.scatter_(1, sender_outcome.action[0].detach(), 1).narrow(1,1,self.base_alphabet_size).float()
                    # summation along batch dimension, and add to counts
                    self._state['current_dist'] += torch.einsum('bi,bj->ij', many_hots, batch.original_category.float().to(self.device)).detach().float()

            if self.log_entropy:
                new_messages = sender_outcome.action[0].view(-1)
                valid_indices = torch.arange(self.base_alphabet_size).expand(new_messages.size(0), self.base_alphabet_size).to(self.device)
                selected_symbols = valid_indices == new_messages.unsqueeze(1).float()
                self._state['symbol_counts'] += selected_symbols.sum(dim=0)

            if self.log_lang_progress and index % 100 == 0:
                if self._state['past_dist'] is None:
                    self._state['past_dist'], self._state['current_dist'] = self._state['current_dist'], torch.zeros((self.base_alphabet_size, 5), dtype=torch.float).to(self.device)
                else:
                    logit_c = (current_dist.view(1, -1) / current_dist.sum()).log()
                    prev_p = (past_dist.view(1, -1) / past_dist.sum())
                    kl = F.kl_div(logit_c, prev_p, reduction='batchmean').item()
                    self._write('llp/kl_div', kl, number_ex_seen, direct=True)
                    self._state['past_dist'], self._state['current_dist'] = self._state['current_dist'], torch.zeros((self.base_alphabet_size, 5), dtype=torch.float).to(self.device)

            if self.log_debug:
                self.log_grads_tensorboard(list(supplementary_info['parameters']))

        self._pbar.update(S=self._state['running_avg_success'])

        return {'running_avg_success': self._state['running_avg_success']}

    def log_grads_tensorboard(self, parameter_list):
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
        self._write('grad/median_grad', median_grad, number_ex_seen)
        self._write('grad/mean_grad', mean_grad, number_ex_seen)
        self._write('grad/max_grad', max_grad, number_ex_seen)
        self._write('grad/mean_norm_grad', mean_norm_grad, number_ex_seen)
        self._write('grad/max_norm_grad', max_norm_grad, number_ex_seen)
