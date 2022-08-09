import argparse, collections, csv, pathlib
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import tqdm
	
base_dir = pathlib.Path(__file__).parents[2] / 'runs'
with open(f'values.csv', 'w') as ostr:
	writer = csv.writer(ostr)
	header = ['epoch', 'max_success_rate'] + [
		f'{tag} from {idx} ({type})' 
		for tag in ('Jac', 'Lev')
		for idx in (0, 50, 75) 
		for type in ('min', 'max', 'mean', 'median', 'std')
	]
	_ = writer.writerow(header)

	for run_file in tqdm.tqdm(list(base_dir.glob('**/event*'))):
		row = [str(run_file)]
		ea = event_accumulator.EventAccumulator(str(run_file))
		ea.Reload()
		max_sr = max(scal.value for scal in ea.Scalars('eval/success_rate'))
		row.append(max_sr)
		for measure_tag in ['FM_corr/Jaccard-based comp', 'FM_corr/Normalised Lev-based comp']:
			try:
				all_scalars = [scal for scal in ea.Scalars(measure_tag)]
			except KeyError:
				# key not found in reservoir
				row.extend([''] * 5 * 3)
			else:
				for starting_step in [0, 50, 75]:
					obs = np.array([scal.value for scal in all_scalars if scal.step >= starting_step])
					if len(obs) > 0:
						row.extend([obs.min(), obs.max(), obs.mean(), np.median(obs), obs.std()])
					else:
						row.extend([''] * 5)
		assert len(row) == len(header)
		_ = writer.writerow(row)
