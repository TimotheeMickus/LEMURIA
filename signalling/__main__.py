#!/usr/bin/env python

from datetime import datetime
import sys

import argparse
import os
import pathlib
import pprint
import sys

#this_path = os.path.abspath(os.path.dirname(sys.argv[0])) # The path of (the directory in which is) this file

import socket # for `gethostname`
import torch # for device

def get_args():
    arg_parser = argparse.ArgumentParser()
    
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
    if not args.quiet:
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
