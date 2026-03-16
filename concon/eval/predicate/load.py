import os, pathlib
import json
import pandas as pd

def get_datapoints(directory_name: str = None):
    '''
    Input:
    - directory_name (string): folder found in 'concon/runs'
    Output:
    - datapoints (list[dict]) of runs with keys: 
        - config (dict), 
        - evaluation (DataFrame), 
        - predicates (DataFrame),
        - languages (list[dict[epoch_number: int, DataFrame]])
    '''
    datapoints = []
    super_directory = pathlib.Path('runs') / directory_name

    for directory in super_directory:
        # Depending on the run, we have several message dumps: store all
        messages_paths = []
        eval_path = None

        # Pick up different CSVs: signals, evaluation, per-predicate metrics.
        for filename in os.listdir(directory):
            if filename.startswith('msgs'):
                messages_paths.append(os.path.join(directory, filename))
            if filename.startswith('eval'):
                eval_path = os.path.join(directory, filename)
            if filename.startswith('predicate'):
                pred_path = os.path.join(directory, filename)

        datapoint = {}
        with open(os.path.join(directory, 'hparams.json')) as f:
            datapoint['config'] = json.load(f)
        datapoint['evaluation'] = pd.read_csv(eval_path)
        datapoint['predicates'] = pd.read_csv(pred_path)
        datapoint['languages'] = []
        for msg_path in messages_paths:
            datapoint['languages'].append({'epoch_number': int(msg_path.split('.')[1].split('e')[1]),
                                           'language': pd.read_csv(msg_path)})

        datapoints.append(datapoint)

    return datapoints