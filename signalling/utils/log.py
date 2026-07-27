from abc import ABCMeta, abstractmethod

import hashlib
import pathlib

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import tqdm

from .misc import compute_entropy


_ALLOWED_NAME_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.="
)

# Arguments left out of the run name by default: paths, environment-specific
# settings, pure IO/logging switches, and the run *count* (the per-run index is
# appended separately as `run=`). Set this to an empty collection to include
# every (declared) argument.
DEFAULT_EXCLUDED_ARGS = frozenset({
    "summary", "models",        # paths (contain '[now]', a hostname and slashes)
    "device",                   # environment-specific
    "wandb", "wandb_project",   # logging destination
    "runs",                     # run count; the per-run index is appended as run=
    "name_with",                # the naming selector itself
    "quiet", "display",         # console verbosity
})

# Hand-picked short names. These pin the abbreviations of the important knobs so
# they stay readable and *stable* no matter what other arguments exist, and they
# resolve every real initials-collision in the current argument set
# (properties/population, max_depth/min_depth, num_candidates/no_conjunction,
# no_summary/no_spigot, ...). Anything not listed here falls back to automatic,
# collision-free initials (see `_auto_abbreviations`).
DEFAULT_ABBREVIATIONS = {
    "properties": "props",        "population": "pop",
    "max_depth": "maxd",          "min_depth": "mind",
    "num_candidates": "cand",     "no_conjunction": "noconj",
    "no_summary": "nosum",        "no_spigot": "nospig",
    "nontrivial_only": "ntriv",   "overfit": "ofit",
    "len_penalty": "lenpen",      "voc_penalty": "vocpen",
    "logging_period": "logper",   "base_alphabet_size": "asize",
    "max_len": "mlen",            "learning_rate": "lr",
    "candidate_encoder": "enc",   "batch_size": "bs",
    "steps_per_epoch": "spe",     "epochs": "ep",
    "reaper_step": "reap",        "pop_size": "pop",
    "pop_reset_period": "preset",
    "jaccard": "jac",
}


def _sanitize_run_part(value):
    text = str(value).strip().replace(" ", "")
    return "".join((c if (c in _ALLOWED_NAME_CHARS) else "-") for c in text)


def _initials(name):
    return "".join(word[0] for word in name.split("_") if word)


def _auto_abbreviations(names, reserved=()):
    """Maps each name to a short, deterministic, globally-unique abbreviation.

    The preferred abbreviation is the initials of the underscore-separated words
    (e.g. `steps_per_epoch` -> `spe`). If that token is already taken (by another
    argument or by a `reserved` value), the name is lengthened one letter per
    word at a time (e.g. `max_depth`/`min_depth` -> `maxd`/`mind`), falling back
    to a numeric suffix only in the pathological case of identical names.

    Names are processed in sorted order so the result is deterministic; the
    `reserved` set (the pinned override abbreviations) is never produced here, so
    auto and override tokens can't collide.
    """
    def candidate(name, k):
        return "".join(word[:k] for word in name.split("_") if word)

    used = set(reserved)
    result = {}
    for name in sorted(names):
        squished = name.replace("_", "")
        k, token = 1, candidate(name, 1)
        while token in used and k < len(squished):
            k += 1
            token = candidate(name, k)
        if token in used:  # exhausted the letters: disambiguate with a suffix
            base, i = token, 0
            while token in used:
                i += 1
                token = f"{base}{i}"
        result[name] = token
        used.add(token)
    return result


def _format_run_value(value):
    if isinstance(value, bool): return "1" if value else "0"      # store_true flags
    if value is None: return "none"
    if isinstance(value, float): return ("%g" % value)            # trims trailing zeros
    if isinstance(value, (list, tuple)):
        return "-".join(_sanitize_run_part(v) for v in value)
    if isinstance(value, pathlib.Path): return _sanitize_run_part(value.name)
    return _sanitize_run_part(value)


def _truncate_with_hash(name, run_index, max_length):
    """Keeps `name` within `max_length` (the filesystem path-component / W&B
    limit) by replacing the overflow with a short, stable hash."""
    if len(name) <= max_length:
        return name
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    suffix = f"__h={digest}__run={run_index}"
    return name[: max(0, max_length - len(suffix))].rstrip("_") + suffix


def _normalize_name_with(name_with):
    """Normalizes the `--name_with` value into a de-duplicated, ordered list of
    argument names. Accepts a list of names, or a single string using brackets,
    commas and/or spaces (e.g. "[min_depth, max_depth]") -- all equivalent."""
    if name_with is None:
        return None
    if isinstance(name_with, str):
        name_with = [name_with]
    seen, names = set(), []
    for item in name_with:
        for part in str(item).replace("[", " ").replace("]", " ").replace(",", " ").split():
            if part and part not in seen:
                seen.add(part)
                names.append(part)
    return names


def build_run_name(args, run_index, name_with=None, defaults=None,
                   excluded=DEFAULT_EXCLUDED_ARGS, overrides=DEFAULT_ABBREVIATIONS,
                   max_length=200):
    """Builds a run name from the arguments.

    If `name_with` is given (a list of argument names, or a bracket/comma/space
    string like "[min_depth, max_depth]"), the name lists exactly those
    arguments, in the given order, as `abbrev=value` tokens joined by `__`,
    followed by the per-launch `t=<run_tag>` timestamp and `run=<run_index>`
    -- e.g. `mind=1__maxd=3__t=2026-07-24_12-00-00_ab12cd__run=0`. Abbreviations
    come from the same scheme as the automatic namer (see `_auto_abbreviations`
    and `overrides`). This mode is explicit, so the listed arguments are always
    included regardless of their default value.

    Otherwise the name is generated automatically: each argument is abbreviated
    to a short token and formatted as `abbrev=value`; tokens are joined by `__`,
    and `run=<run_index>` is appended last. `args` may be an argparse.Namespace
    or a plain dict.

    If `defaults` (a name -> default-value mapping, e.g. the argument parser's
    defaults) is provided, only arguments that (a) are declared in `defaults`
    and (b) differ from their default are included. This is the recommended
    automatic mode: a run advertises only what it actually changed, so names
    stay short and legible. When `defaults` is None, every non-excluded declared
    argument is included, and the result is hash-truncated if it would exceed
    `max_length` (which also keeps it a legal filesystem path component).

    `excluded` names to drop entirely; `overrides` pins chosen abbreviations.
    `run_tag`, private (`_`-prefixed) attributes, and anything absent from
    `defaults` (e.g. values injected into `args` after parsing) are never swept
    into the automatic name; `run_tag`, if present and truthy, is appended as a
    `t=` suffix there.
    """
    values = args if isinstance(args, dict) else vars(args)

    selected = _normalize_name_with(name_with)
    if selected:
        unknown = [n for n in selected if n not in values]
        if unknown:
            available = ", ".join(sorted(k for k in values if not k.startswith("_")))
            raise KeyError(
                f"--name_with references unknown argument(s): {', '.join(unknown)}. "
                f"Available names: {available}."
            )
        reserved = {overrides[n] for n in selected if n in overrides}
        auto = _auto_abbreviations([n for n in selected if n not in overrides], reserved=reserved)
        abbr = {n: overrides.get(n, auto.get(n)) for n in selected}
        parts = [f"{abbr[n]}={_format_run_value(values[n])}" for n in selected]
        run_tag = values.get("run_tag")
        if run_tag and "run_tag" not in selected:  # per-launch timestamp
            parts.append(f"t={_sanitize_run_part(run_tag)}")
        parts.append(f"run={run_index}")
        return _truncate_with_hash("__".join(parts), run_index, max_length)

    def _keep(key, value):
        if key.startswith("_") or key == "run_tag" or key in excluded:
            return False
        if defaults is None:
            return True
        if key not in defaults:  # e.g. attributes injected after parsing
            return False
        return value != defaults[key]

    items = sorted((k, v) for k, v in values.items() if _keep(k, v))
    names = [k for k, _ in items]
    reserved = {overrides[n] for n in names if n in overrides}
    auto = _auto_abbreviations([n for n in names if n not in overrides], reserved=reserved)
    abbr = {n: overrides.get(n, auto.get(n)) for n in names}

    parts = [f"{abbr[k]}={_format_run_value(v)}" for k, v in items]

    run_tag = values.get("run_tag")
    if run_tag:
        parts.append(f"t={_sanitize_run_part(run_tag)}")
    parts.append(f"run={run_index}")

    name = "__".join(parts)
    return _truncate_with_hash(name, run_index, max_length)


def _wandb_config_from_args(args):
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, torch.device):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def setup_wandb_logging(autologger, enabled, project, run_name, args):
    if not enabled:
        return None

    import wandb

    wandb_run = wandb.init(project=project, name=run_name, config=_wandb_config_from_args(args))
    write_to_summary = autologger._write

    def _write_and_wandb(name, value, step, direct=False):
        write_to_summary(name, value, step, direct=direct)
        # Train logs pass an iteration-like step, while eval logs pass an epoch index with direct=True.
        # Convert eval steps to the scale so both curves share one WandB x-axis.
        if direct: wandb_step = (int(step) + 1) * int(getattr(args, "steps_per_epoch", 1))
        else: wandb_step = int(step)
        wandb.log({name: value}, step=wandb_step)
    autologger._write = _write_and_wandb
    return wandb_run


def finish_wandb_logging(wandb_run):
    if wandb_run is not None:
        wandb_run.finish()

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

        if len(external_output) == 1 and isinstance(external_output[0], dict):
            metrics = external_output[0]
            rewards = metrics["rewards"]
            successes = metrics["successes"]
            signal_length = metrics["signal_length"]
            sender_entropy = metrics["sender_entropy"]
            receiver_entropy = metrics["receiver_entropy"]
        else:
            rewards, successes, signal_length, sender_entropy, receiver_entropy, *external_output = external_output

        # Computes the minimum length the signals can have in order to get perfect accuracy (approximation when the size of the alphabet >> 1)
        minimal_compression_len = np.log(self.data_loader.nb_categories) / np.log(self.base_alphabet_size + 1) # + 1 because EoM is taken into account
        length_ratio = (signal_length / minimal_compression_len).item()

        # updates running average reward
        self._state['total_reward'] += rewards.sum().item()
        self._state['total_success'] += successes.sum().item()
        self._state['total_items'] += supplementary_info['batch'].size
        self._state['running_avg_reward'] = self._state['total_reward'] / self._state['total_items']
        self._state['running_avg_success'] = self._state['total_success'] / self._state['total_items']

        if self.summary_writer is not None:
            avg_reward = rewards.mean().item() # average reward of the batch
            avg_success = successes.mean().item() # average success of the batch
            number_ex_seen = supplementary_info['index']
            self._state['number_ex_seen'] = number_ex_seen

            self._write('train/reward', avg_reward, number_ex_seen)
            self._write('train/success', avg_success, number_ex_seen)
            self._write('train/loss', loss.item(), number_ex_seen)
            self._write('train/sender_entropy', sender_entropy.item(), number_ex_seen)
            self._write('train/receiver_entropy', receiver_entropy.item(), number_ex_seen)

            self._write('llp/signal_length', signal_length.item(), number_ex_seen)
            self._write('llp/length_ratio', length_ratio, number_ex_seen)

            if self.log_lang_progress:
                for batch in supplementary_info['batches']:
                    batch_signal_manyhot = torch.zeros((batch.size, self.base_alphabet_size + 2), dtype=torch.float).to(self.device) # size of embeddings + EOS + PAD
                    # signal -> many-hot
                    many_hots = batch_signal_manyhot.scatter_(1, sender_outcome.action[0].detach(), 1).narrow(1,1,self.base_alphabet_size).float()
                    # summation along batch dimension, and add to counts
                    self._state['current_dist'] += torch.einsum('bi,bj->ij', many_hots, batch.original_category.float().to(self.device)).detach().float()

            if self.log_entropy:
                new_signals = sender_outcome.action[0].view(-1)
                valid_indices = torch.arange(self.base_alphabet_size).expand(new_signals.size(0), self.base_alphabet_size).to(self.device)
                selected_symbols = valid_indices == new_signals.unsqueeze(1).float()
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
