#!/usr/bin/env python

from datetime import datetime
import sys

import argparse
import os
import pathlib
import pprint
import sys

#this_path = os.path.abspath(os.path.dirname(sys.argv[0])) # The path of (the directory in which is) this file

# The subcommands that own a full argument parser of their own (built in their get_args). For these,
# `--help` is handed through so that the *game's* arguments are shown; every other subcommand's
# arguments live on this top-level parser, so its help is shown instead.
GAME_SUBCOMMANDS = {'image_signalling_game', 'predicate_signalling_game'}

def get_args():
    # add_help=False: we register --help ourselves (below) so that its eager argparse action does not
    # fire during parse_known_args and pre-empt the per-game `--do <game> --help`.
    arg_parser = argparse.ArgumentParser(add_help=False)

    arg_parser.add_argument('-h', '--help', action='store_true', help="show this help message and exit (with --do <game>, shows that game's own arguments)")

    arg_parser.add_argument('--do', help='what to do', type=str, choices=['image_signalling_game', 'predicate_signalling_game', 'evaluate_language', 'visualize', 'compute_correlation', 'threeway_correlation', 'compositionality_search'])
    
    group = arg_parser.add_argument_group(title='Display', description='arguments relative to displayed information')
    # TODO: refactor logging: --quiet vs. --display quiet?
    group.add_argument('--quiet', help='display less information', action='store_true')
    
    group = arg_parser.add_argument_group(title='Eval', description='arguments relative to evaluation routines')
    group.add_argument('--analysis_gram_size', help='size of the n-grams considered during language analysis', type=int, default=1)
    group.add_argument('--analysis_disj_size', help='size of the disjunctions considered during language analysis', type=int, default=1)

    # TODO: assuming subcommands, these should end up in relevant subparsers
    # For visualize.py / evaluate_language.py
    group.add_argument('--load_model', help='the path to the model to load', type=pathlib.Path)
    # For evaluate_language.py
    group.add_argument('--load_other_model', help='path to a second model to load', type=pathlib.Path)
    group.add_argument('--string_signals', action='store_true', help='specifies whether provided signals should be considered as strings rather than sequences of symbols (integers).')
    
    group.add_argument('--debug', '-d', help='use this flag to change the behavior of the code to debug stuff', action='store_true')


    #args = arg_parser.parse_args()
    (args, remaining_args) = arg_parser.parse_known_args()

    if(args.help):
        if(args.do in GAME_SUBCOMMANDS):
            # Re-inject --help so the selected game's parser renders its own (sectioned) help.
            remaining_args = list(remaining_args) + ['--help']
        else:
            arg_parser.print_help()
            arg_parser.exit()

    if(not args.quiet and not args.help):
        print("command-line arguments:")
        pprint.pprint(vars(args), indent=4)

    return (args, remaining_args)



if(__name__ == "__main__"):
    args, remaining_args = get_args()
    if(args.do == 'evaluate_language'):
        from .eval.evaluate_language import main
        main(args) # maybe switch to main(args, remaining_args)
    elif(args.do == 'visualize'):
        from .eval.visualize import main
        main(args) # maybe switch to main(args, remaining_args)
    elif(args.do == 'compute_correlation'):
        from .eval.compute_correlation import main
        main(args) # maybe switch to main(args, remaining_args)
    elif(args.do == 'threeway_correlation'):
        from .eval.three_way_correlation import main
        main(args) # maybe switch to main(args, remaining_args)
    elif(args.do == 'image_signalling_game'):
        from .image_signalling_game import main
        main(args, remaining_args)
    elif(args.do == 'predicate_signalling_game'):
        from .predicate_signalling_game import main
        main(args, remaining_args)
    elif(args.do == 'compositionality_search'):
        from .eval.compositionality import main
        main(args, remaining_args)
    else:
        print(f'I do not know what to do ("{args.do}").')
