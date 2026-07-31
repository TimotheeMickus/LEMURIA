"""Shared command-line argument helpers.

Each function adds one `argparse` group to a parser and returns that group, so a
caller can append a few game-specific arguments to the same visual section (this
keeps `--help` sectioned). These cover the infrastructure arguments that are
genuinely common to both signalling games; arguments specific to a dataset, to
an encoder, to pretraining, to populations, etc. are defined next to the code
that uses them (see the corresponding modules).
"""

import pathlib


def add_display_args(parser, image_logging=False):
    """Display/logging switches. `image_logging=True` adds the image-only knobs."""
    group = parser.add_argument_group(title='Display', description='arguments relative to displayed information')
    # TODO: refactor logging: --display tqdm should be inferred from the env
    group.add_argument('--display', help='how to display the information', choices=['minimal', 'simple', 'tqdm'], default='tqdm')
    group.add_argument('--log_debug', '-ld', help='log more stuf', action='store_true')
    group.add_argument('--detect_anomaly', help='autodetect grad anomalies', action='store_true')
    group.add_argument('--no_summary', '-ns', help='do not write summaries', action='store_true')
    group.add_argument('--log_lang_progress', '-llp', help='log metrics to evaluate progress and stability of language learned', action='store_true')
    group.add_argument('--log_entropy', help='log evolution of entropy across epochs', action='store_true')
    if image_logging:
        group.add_argument('--no_log_imgs', help='do not log image samples', action='store_true')
        group.add_argument('--log_img_every', default=10, type=int, help='how often (in epochs) to log image samples')
    # TODO: refactor logging: --logging_period should control the frequency of step reports when --display minimal
    group.add_argument('--logging_period', help='how often counts of logged variables are accumulated', type=int, default=10)
    # TODO: refactor logging: --quiet vs. --display quiet?
    group.add_argument('--quiet', help='display less information', action='store_true')
    return group


def add_save_args(parser, default_summary, default_models):
    """Where to write summaries/models. Callers may add game-specific dump knobs."""
    group = parser.add_argument_group(title='Save', description='arguments relative to saving models/logs')
    group.add_argument('--summary', help='the path to the TensorBoard summary for this run (\'[now]\' will be intepreted as now in the Y-m-d_H-M-S format)', default=default_summary, type=pathlib.Path)
    group.add_argument('--save_every', '-save_every', help='indicate to save the model after each __ epochs', type=int, default=0)
    group.add_argument('--models', help='the path to the saved models (\'[summary]\' will be interpreted as the value of --summary)', default=default_models, type=pathlib.Path)
    return group


def add_reward_args(parser, len_penalty_default):
    """Reward shaping / gradient computation shared by both games. `len_penalty`'s
    default differs per game, hence the required parameter. Entropy betas are named
    differently per game (sender/receiver vs asker/retriever), so callers add those."""
    group = parser.add_argument_group(title='Reward', description='arguments relative to reward shaping/gradient computation')
    group.add_argument('--len_penalty', help='coefficient for the length penalty of the signals', default=len_penalty_default, type=float)
    group.add_argument('--use_expectation', help='use expectation of success instead of playing dice', action='store_true')
    group.add_argument("--learning_rate", help="learning rate", default=0.0001, type=float)
    group.add_argument("--learning_rate_a", help="learning rate for the sender/asker (overrides --learning_rate if set)", default=None, type=float)
    group.add_argument("--learning_rate_b", help="learning rate for the receiver/retriever (overrides --learning_rate if set)", default=None, type=float)
    group.add_argument('--grad_clipping', help='threshold for gradient clipping', default=1, type=float)
    group.add_argument('--grad_scaling', help='threshold for gradient scaling', default=None, type=float)
    return group


def add_language_args(parser):
    group = parser.add_argument_group(title='Language', description='arguments relative to language capacity')
    group.add_argument('--base_alphabet_size', help='size of the alphabet (not including special symbols)', default=10, type=int)
    group.add_argument('--max_len', help='maximum length of signals produced', default=10, type=int)
    return group


def add_perf_args(parser):
    import torch
    group = parser.add_argument_group(title='Perfs', description='arguments relative to performances')
    group.add_argument('--device', help='what to run PyTorch on', type=torch.device, default=torch.device('cpu'))
    return group


def add_training_args(parser):
    """Core training-loop knobs shared by both games. Callers may append
    game-specific training arguments (e.g. --keep_training) to the returned group."""
    group = parser.add_argument_group(title='Training', description='arguments relative to training')
    group.add_argument('--use_baseline', help='use a baseline term in REINFORCE', action='store_true')
    group.add_argument('--epochs', help='number of epochs', default=100, type=int)
    group.add_argument('--steps_per_epoch', help='number of steps per epoch', default=1000, type=int)
    group.add_argument('--runs', help='number of runs', default=1, type=int)
    group.add_argument('--seed', help='base random seed; each run uses seed+run_idx', default=None, type=int)
    group.add_argument('--loss_weight_temp', help='temperature parameter in the loss weighting system', default=1.0, type=float)
    return group


def add_debug_arg(parser):
    """The single --debug flag (kept identical across games)."""
    parser.add_argument('--debug', '-d', help='use this flag to change the behavior of the code to debug stuff', action='store_true')
