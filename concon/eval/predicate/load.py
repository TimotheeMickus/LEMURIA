import os, pathlib
import json
import pandas as pd
from tqdm import tqdm

def get_datapoints(directory_name: str = None, latest_language_only: bool = False):
    '''
    List-returning wrapper around iter_datapoints.
    Input:
    - directory_name (string): folder found in 'concon/runs'
    Output:
    - datapoints (list[dict]) of runs with keys:
        - config (dict), 
        - evaluation (DataFrame), 
        - predicates (DataFrame),
        - languages (list[dict[epoch_number: int, DataFrame]]),
        - folder_name (str),
        - run_name (str)
    '''
    return list(iter_datapoints(directory_name, latest_language_only=latest_language_only))


def iter_datapoints(directory_name: str = None, latest_language_only: bool = False):
    '''
    Input:
    - directory_name (string): folder found in 'concon/runs'
    Output:
    - yields datapoints (dict) with keys:
        - config (dict), 
        - evaluation (DataFrame), 
        - predicates (DataFrame),
        - languages (list[dict[epoch_number: int, DataFrame]]),
        - folder_name (str),
        - run_name (str)
    '''
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    super_directory = repo_root / "runs" / directory_name

    directories = list(super_directory.iterdir())
    for directory in tqdm(directories, desc="Loading datapoints"):
        if not directory.is_dir():
            continue
        # Handle cases where directories are nested
        if not (directory / "hparams.json").is_file():
            subdirectories = [p for p in directory.iterdir() if p.is_dir()]
            if len(subdirectories) == 1 and (subdirectories[0] / "hparams.json").is_file():
                directory = subdirectories[0]
            else:
                print(f"Found weird directory in {super_directory}/{directory}.\n{subdirectories}.")
        # Depending on the run, we have several message dumps: store all
        messages_paths = []
        eval_path = None
        pred_path = None

        # Pick up different CSVs: signals, evaluation, per-predicate metrics.
        for filename in os.listdir(directory):
            if filename.startswith('msgs'):
                messages_paths.append((os.path.join(directory, filename), filename))
            if filename.startswith('eval'):
                eval_path = os.path.join(directory, filename)
            if filename.startswith('predicate'):
                pred_path = os.path.join(directory, filename)

        datapoint = {}
        datapoint['folder_name'] = directory.name
        datapoint['run_name'] = directory.name
        with open(os.path.join(directory, 'hparams.json')) as f:
            datapoint['config'] = json.load(f)

        datapoint['evaluation'] = pd.read_csv(eval_path) if eval_path is not None else None
        datapoint['predicates'] = pd.read_csv(pred_path) if pred_path is not None else None
        datapoint['languages'] = []
        if latest_language_only and messages_paths:
            msg_path, fname = max(messages_paths, key=lambda x: int(x[1].split('.')[1].split('e')[1]))
            datapoint['languages'].append({
                'epoch_number': int(fname.split('.')[1].split('e')[1]),
                'language': pd.read_csv(msg_path),
            })
        else:
            for msg_path, fname in messages_paths:
                datapoint['languages'].append({'epoch_number': int(fname.split('.')[1].split('e')[1]),
                                            'language': pd.read_csv(msg_path)})

        yield datapoint
