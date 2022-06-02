import collections, csv, pathlib
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def parse_single_exp(exp_dir, *tags):
	steps_avg = {tag: collections.defaultdict(list) for tag in tags}
	for run in exp_dir.glob('**/event*'):
		ea = event_accumulator.EventAccumulator(str(run))
		ea.Reload()
		for tag in tags:
			for scal in ea.Scalars(tag):
				steps_avg[tag][scal.step].append(scal.value)
	return {tag: {step : np.array(vals) for step, vals in steps_avg[tag].items()} for tag in steps_avg}
	
base_dir = pathlib.Path('runs/pretrain-search')
for mode in ('category-wise', 'feature-wise', 'auto-encoder'):
	lr_dirs = (base_dir / mode).iterdir()
	lr_maps = {}
	tags =  f'eval-pretrain/loss_agent 0_{mode}', f'eval-pretrain/acc_agent 0_{mode}'
	if mode == 'auto-encoder':
		tags = f'eval-pretrain/loss_agent 0_{mode}',
	for exp_dir in lr_dirs:
		lr_maps[exp_dir.name] = parse_single_exp(exp_dir, *tags)
	with open(f'{mode}.csv', 'w') as ostr:
		writer = csv.writer(ostr)
		sorted_lr_keys = sorted(lr_maps.keys())
		header = [
			'epoch', 
			*(
				f'loss_{lr}_mean'.replace('_','').replace(' ', '') 
				for lr in sorted_lr_keys
			), 
			*(
				f'loss_{lr}_std'.replace('_','').replace(' ', '') 
				for lr in sorted_lr_keys
			), 
		]
		if mode != 'auto-encoder':
			header += [
				*(
					f'acc_{lr}_mean'.replace('_','').replace(' ', '') 
					for lr in sorted_lr_keys
				), 
				*(
					f'acc_{lr}_std'.replace('_','').replace(' ', '') 
					for lr in sorted_lr_keys
				),
			]
		_ = writer.writerow(header)
		loss_k = f'eval-pretrain/loss_agent 0_{mode}'
		acc_k = f'eval-pretrain/acc_agent 0_{mode}'
		for epoch in range(10):
			row = [epoch]
			row +=	[
				lr_maps[lr][loss_k][epoch].mean() 
				for lr in sorted_lr_keys
			]
			row +=	[
				lr_maps[lr][loss_k][epoch].std() 
				for lr in sorted_lr_keys
			]
			if mode != 'auto-encoder':
				row +=	[
					lr_maps[lr][acc_k][epoch].mean() 
					for lr in sorted_lr_keys
				]
				row +=	[
					lr_maps[lr][acc_k][epoch].std() 
					for lr in sorted_lr_keys
				]
			_ = writer.writerow(row)
