import argparse, collections, csv, pathlib
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
	'--runs_dir', 
	type=pathlib.Path, 
	default=pathlib.Path(__file__).absolute().parents[2] / 'runs'
)
parser.add_argument('--tags', nargs='+', type=str, required=True)
parser.add_argument('--alt-tags', nargs='*', type=str)
parser.add_argument('--steps', nargs='+', type=int, default=[0, 50, 75])
parser.add_argument('--output-file', type=pathlib.Path, default='logs-summary.csv')
args = parser.parse_args()
assert all((alt_tag in args.tags) for alt_tag in args.alt_tags), 'Incoherent specs'

base_dir = args.runs_dir
with open(args.output_file, 'w') as ostr:
	writer = csv.writer(ostr)
	header = ['event_file'] + [
		f'{tag} from {idx} ({type})'
		for tag in args.tags
		for idx in args.steps
		for type in ('min', 'max', 'mean', 'median', 'std')
	] + [
		f'ALT {tag} from {idx} ({type})'
		for tag in args.alt_tags
		for idx in args.steps
		for type in ('min', 'max', 'mean', 'median', 'std')
	]
	_ = writer.writerow(header)

	for run_file in tqdm.tqdm(list(base_dir.glob('**/event*'))):
		row = [str(run_file)]
		ea = event_accumulator.EventAccumulator(str(run_file))
		ea.Reload()
		for measure_tag in args.tags:
			try:
				all_scalars = [scal for scal in ea.Scalars(measure_tag)]
			except KeyError:
				# key not found in reservoir
				row.extend([''] * 5 * 3)
			else:
				for starting_step in args.steps:
					obs = np.array([scal.value for scal in all_scalars if scal.step >= starting_step])
					if len(obs) > 0:
						row.extend([obs.min(), obs.max(), obs.mean(), np.median(obs), obs.std()])
					else:
						row.extend([''] * 5)
		for measure_tag in args.alt_tags:
			try:
				all_scalars = [scal for scal in ea.Scalars(measure_tag)]
			except KeyError:
				# key not found in reservoir
				row.extend([''] * 5 * 3)
			else:
				for starting_step in args.steps:
					obs = np.array([-np.log(1 - scal.value) for scal in all_scalars if scal.step >= starting_step])
					if len(obs) > 0:
						row.extend([obs.min(), obs.max(), obs.mean(), np.median(obs), obs.std()])
					else:
						row.extend([''] * 5)
		assert len(row) == len(header)
		_ = writer.writerow(row)
