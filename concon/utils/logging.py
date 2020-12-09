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
            _tag = tag if(self.prefix is None) else (self.prefix + ':' +  tag)
            self.writer.add_scalar(tag=_tag, scalar_value=np.mean(values), global_step=global_step)

            values.clear()

    # `l` is a list of pairs (key, value)
    def add_scalar_list(self, l, global_step=None):
        add = False
        for key, value in l:
            self.add_scalar(key, value, global_step)

# TODO Instead of having one class Progress with a `display` parameter, it might be more elegant to have different classes. The AutoLogger would chose which one to use dependeing on the `display` parameter.
class Progress:
    def __init__(self, display, steps_per_epoch, epoch, logged_items={"R"}):
        self.display = display
        self.steps_per_epoch = steps_per_epoch
        self.epoch = epoch
        self._logged_items = logged_items

    def __enter__(self):
        if(self.display == 'normal'): self.i = 0
        elif(self.display == 'tqdm'): self.pbar = tqdm.tqdm(total=self.steps_per_epoch, postfix={i: 0.0 for i in self._logged_items}, unit="B", desc=("Epoch %i" % self.epoch)) # Do not forget to close it at the end

        return self

    def update(self, **logged_items):
        if(self.display == 'normal'):
            postfix = " ".join(("%s: %f" % (k, logged_items[k])) for k in sorted(logged_items))
            print(('%i/%i - %s' % (self.i, self.steps_per_epoch, postfix)), flush=True)
            self.i += 1
        elif(self.display == 'tqdm'):
            self.pbar.set_postfix(logged_items, refresh=False)
            self.pbar.update()

    def __exit__(self, type, value, traceback):
        if(self.display == 'tqdm'): self.pbar.close()

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

# TODO Could have a 'print(self, message, flush, type)' method
class AutoLogger(object):
    def __init__(self, game, data_loader, display='tqdm', steps_per_epoch=1000, debug=False,
        log_lang_progress=False, log_entropy=False,
        device='cpu', no_summary=False, summary_dir=None, default_period=10,):
        #log_charlie_acc=False):

        self.game = game
        self.data_loader = data_loader

        self.display = display
        self.steps_per_epoch = steps_per_epoch

        self.current_epoch = 0
        self.global_step = 0
        self.logged_items = {"S"}

        self.log_debug = debug
        self.log_lang_progress = log_lang_progress
        self.log_entropy = log_entropy

        self.device = device

        #self.log_charlie_acc = log_charlie_acc # Commented out: use self.tag_header to highlight when results pertain to Charlie

        if no_summary:
            self.summary_writer = None
        else:
            self.summary_writer = AverageSummaryWriter(log_dir=summary_dir, default_period=default_period)
        self._pbar = None

        self._state = {}

        self.tag_header = ""

    def new_progress_bar(self):
        self._pbar = Progress(self.display, self.steps_per_epoch, self.current_epoch, self.logged_items)
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
                self._state['current_dist'] = torch.zeros((self.game.base_alphabet_size, 5), dtype=torch.float).to(self.device)
                self._state['past_dist'] = None # size of embeddings past_dist, current_dist = None,

            if self.log_entropy:
                self._state['symbol_counts'] = torch.zeros(self.game.base_alphabet_size, dtype=torch.float).to(self.device)

        self._pbar.__enter__()

        return self

    def _write(self, tag, val, step, direct=False):
        """Convenience method for writing to summary. Prepends with tag_header on the fly if necessary.
            args:
                tag, val, step: for summary
            direct: target summary_writer.writer (wrapped object) instead of summary_writer
        """
        writer = self.summary_writer.writer if direct else self.summary_writer
        if self.tag_header: tag = '%s-%s' % (self.tag_header, tag)
        return writer.add_scalar(tag, val, step)

    def __exit__(self, type, value, traceback):
        if self.log_entropy and self.summary_writer is not None:
            self._write('llp/language_entropy', compute_entropy(self._state['symbol_counts'], base=2), self._state['number_ex_seen'], direct=True)

        self._pbar.__exit__(type, value, traceback)
        self._state = {}


    def update(self, loss, *external_output, **supplementary_info):
        rewards, successes, avg_msg_length, sender_entropy, receiver_entropy, *external_output = external_output

        # Computes the minimum length the messages can have in order to get perfect accuracy (approximation when the size of the alphabet >> 1)
        minimal_compression_len = np.log(self.data_loader.nb_categories) / np.log(self.game.base_alphabet_size + 1) # + 1 because EoM is taken into account
        length_ratio = (avg_msg_length / minimal_compression_len)

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

            self._write('llp/msg_length', avg_msg_length, number_ex_seen)
            self._write('llp/length_ratio', length_ratio, number_ex_seen)

            # if self.log_charlie_acc:
            #     charlie_acc = charlie_acc.mean().item()
            #     self.summary_writer.add_scalar('train-%s/charlie_acc' % tag, charlie_acc, number_ex_seen)

            if self.log_lang_progress:
                for batch in supplementary_info['batches']:
                    batch_msg_manyhot = torch.zeros((batch.size, self.game.base_alphabet_size + 2), dtype=torch.float).to(self.device) # size of embeddings + EOS + PAD
                    # message -> many-hot
                    many_hots = batch_msg_manyhot.scatter_(1, sender_outcome.action[0].detach(), 1).narrow(1,1,self.game.base_alphabet_size).float()
                    # summation along batch dimension, and add to counts
                    self._state['current_dist'] += torch.einsum('bi,bj->ij', many_hots, batch.original_category.float().to(self.device)).detach().float()

            if self.log_entropy:
                new_messages = sender_outcome.action[0].view(-1)
                valid_indices = torch.arange(self.game.base_alphabet_size).expand(new_messages.size(0), self.game.base_alphabet_size).to(self.device)
                selected_symbols = valid_indices == new_messages.unsqueeze(1).float()
                self._state['symbol_counts'] += selected_symbols.sum(dim=0)

            if self.log_lang_progress and index % 100 == 0:
                if self._state['past_dist'] is None:
                    self._state['past_dist'], self._state['current_dist'] = self._state['current_dist'], torch.zeros((self.game.base_alphabet_size, 5), dtype=torch.float).to(self.device)
                else:
                    logit_c = (current_dist.view(1, -1) / current_dist.sum()).log()
                    prev_p = (past_dist.view(1, -1) / past_dist.sum())
                    kl = F.kl_div(logit_c, prev_p, reduction='batchmean').item()
                    self._write('llp/kl_div', kl, number_ex_seen, direct=True)
                    self._state['past_dist'], self._state['current_dist'] = self._state['current_dist'], torch.zeros((self.game.base_alphabet_size, 5), dtype=torch.float).to(self.device)

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
