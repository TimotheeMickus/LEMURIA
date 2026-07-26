import os, pathlib
import json
import pandas as pd
from tqdm import tqdm

def get_datapoints(directory_name: str = None, latest_only: bool = False, load_languages: bool = True):
    '''
    List-returning wrapper around iter_datapoints.
    Input:
    - directory_name (string): folder found in 'concon/runs'
    - latest_only (bool): if True, load only the highest-epoch language dump per run
    - load_languages (bool): if False, skip loading any language/signals files entirely
    Output:
    - datapoints (list[dict]) of runs with keys:
        - config (dict),
        - evaluation (DataFrame),
        - predicates (DataFrame),
        - languages (list[dict[epoch_number: int, DataFrame]])
    '''
    return list(iter_datapoints(directory_name, latest_only=latest_only, load_languages=load_languages))


def iter_datapoints(directory_name: str = None, latest_only: bool = False, load_languages: bool = True):
    '''
    Input:
    - directory_name (string): folder found in 'concon/runs'
    - latest_only (bool): if True, load only the highest-epoch language dump per run
    - load_languages (bool): if False, skip loading any language/signals files entirely
    Output:
    - yields datapoints (dict) with keys:
        - config (dict),
        - evaluation (DataFrame),
        - predicates (DataFrame),
        - languages (list[dict[epoch_number: int, DataFrame]])
    '''
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    super_directory = repo_root / "runs" / directory_name

    directories = list(super_directory.iterdir())
    for directory in tqdm(directories, desc="Loading datapoints"):
        # Handle cases where directories are nested
        if not (directory / "hparams.json").is_file():
            subdirectories = [p for p in directory.iterdir() if p.is_dir()]
            if len(subdirectories) == 1 and (subdirectories[0] / "hparams.json").is_file():
                directory = subdirectories[0]
            else:
                print(f"Found weird directory in {super_directory}/{directory}.\n{subdirectories}.")
        signals_paths = []
        eval_path = None
        pred_path = None

        for filename in os.listdir(directory):
            if load_languages and filename.startswith('signals') and not filename.endswith('.bak'):
                signals_paths.append((os.path.join(directory, filename), filename))
            if filename.startswith('eval'):
                eval_path = os.path.join(directory, filename)
            if load_languages and filename.startswith('predicate'):
                pred_path = os.path.join(directory, filename)

        if latest_only and signals_paths:
            signals_paths = [max(signals_paths, key=lambda x: int(x[1].split('.')[1].split('e')[1]))]

        datapoint = {}
        with open(os.path.join(directory, 'hparams.json')) as f:
            datapoint['config'] = json.load(f)
        datapoint['run_name'] = directory.name
        datapoint['run_path'] = str(directory)

        datapoint['evaluation'] = pd.read_csv(eval_path) if eval_path is not None else None
        datapoint['predicates'] = pd.read_csv(pred_path) if pred_path is not None else None
        datapoint['languages'] = []
        for signal_path, fname in signals_paths:
            datapoint['languages'].append({'epoch_number': int(fname.split('.')[1].split('e')[1]),
                                        'language': pd.read_csv(signal_path)})

        yield datapoint
