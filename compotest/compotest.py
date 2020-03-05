import random
import re
import itertools
import functools
import subprocess
import collections

import numpy as np
import scipy.stats
import Levenshtein
import torch
import multiprocessing

from allennlp.predictors.predictor import Predictor

import argparse
import sys
import json


#### Meaning distances
def cdist(v1, v2):
	return 1.0 - (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def l2(v1, v2) :
	return np.linalg.norm(v1 - v2)


#### Text distances
@functools.lru_cache(maxsize=524288)
def apted(tree1, tree2):
	cmd = 'java -jar apted.jar -t %s %s' % (tree1, tree2)
	apted_output = subprocess.check_output(cmd.split())
	score = float(apted_output.decode("utf-8"))
	return score

@functools.lru_cache(maxsize=524288)
def jaccard(seq1, seq2):
    union = len(seq1)
    intersection = 0
    d = collections.defaultdict(int)
    for i in seq1:
        d[i] += 1
    for i in seq2:
        x = d[i]
        if(x > 0):
            d[i] -= 1
            intersection += 1
        else:
            union += 1
    return 1 - (intersection / union)

@functools.lru_cache(maxsize=524288)
def levenshtein(str1, str2, normalise=False):
    tmp = Levenshtein.distance(str1, str2)
    if(normalise): tmp /= (len(str1) + len(str2))

    return tmp

@functools.lru_cache(maxsize=524288)
def levenshtein_normalised(str1, str2):
    return levenshtein(str1, str2, normalise=True)


#### Misc functions
@functools.lru_cache(maxsize=524288)
def to_brkt(tree):
	"""
	AllenNLP -> apted format
	"""
	prep = tree.replace('(', '{').replace(')', '}')
	prep = re.sub(r" ([^{} ]+)}",r" {\1}}", prep)
	return prep.replace(' ', '')

def sample_pairs(sentences, restrict_dataset_size=None, sample_size=1024):
	sample = range(len(sentences))

	if True or restrict_dataset_size:
		sample = random.sample(sample, 4096)

	sample = list(itertools.combinations(sample, 2))
	#sample = random.sample(sample, sample_size)
	return sample


def correlation_fn(single_arg):
	"""
	Compute correlation of text distance and meaning distance
	"""
	sentences, tree_idx, pos_decored_idx, meanings_idx, w2c, remap, dump_vals = single_arg

	if remap:
		uniq_cats = {i for p in sample for i in p}
		num_cats = len(uniq_cats)
		mapping = dict(zip(uniq_cats, random.sample(uniq_cats, num_cats)))
	vals = []
	for idx1, idx2 in sample:
		if remap:
			meaning_1 = meanings_idx[mapping[idx1]]
			meaning_2 = meanings_idx[mapping[idx2]]
		else:
			meaning_1 = meanings_idx[idx1]
			meaning_2 = meanings_idx[idx2]

		tree1 = tree_idx[idx1]
		tree2 = tree_idx[idx2]

		sentence_1 = sentences[idx1]
		sentence_2 = sentences[idx2]

		sentence_pos_1 = pos_decored_idx[idx1]
		sentence_pos_2 = pos_decored_idx[idx2]


		chars_1 = ''.join(chr(w2c[w]) for w in sentence_1)
		chars_2 = ''.join(chr(w2c[w]) for w in sentence_2)

		chars_pos_1 = ''.join(chr(w2c[w]) for w in sentence_pos_1)
		chars_pos_2 = ''.join(chr(w2c[w]) for w in sentence_pos_2)

		cdist_score = cdist(meaning_1, meaning_2)
		l2_score = l2(meaning_1, meaning_2)

		levenshtein_score = levenshtein(chars_1, chars_2)
		levenshtein_pos_score = levenshtein(chars_pos_1, chars_pos_2)
		levenshtein_n_score = levenshtein_normalised(chars_1, chars_2)
		jaccard_score = jaccard(sentence_1, sentence_2)
		apted_score = apted(tree1, tree2)

		tmp_results = {
			'idx': [idx1, idx2],
			'meaning_scores': {
				'cdist': cdist_score,
				'l2': l2_score,
			},
			'text_scores': {
				'levenshtein': levenshtein_score,
				'levenshtein_pos':levenshtein_pos_score,
				'levenshtein_n': levenshtein_n_score,
				'jaccard': jaccard_score,
				'apted': apted_score,
			},
		}

		vals.append(tmp_results)
	if dump_vals:
		print('dumping scores...')
		with open(dump_vals, 'w') as dumpfile:
			json.dump(vals, dumpfile)
	cdist_scores = [r['meaning_scores']['cdist'] for r in vals]
	l2_scores = [r['meaning_scores']['l2'] for r in vals]

	levenshtein_scores = [r['text_scores']['levenshtein'] for r in vals]
	levenshtein_pos_scores = [r['text_scores']['levenshtein_pos'] for r in vals]
	levenshtein_n_scores = [r['text_scores']['levenshtein_n'] for r in vals]
	jaccard_scores = [r['text_scores']['jaccard'] for r in vals]
	apted_scores = [r['text_scores']['apted'] for r in vals]


	results = {}
	for m_d, m_d_name in ((cdist_scores, 'cdist'), (l2_scores, 'l2')):
		for t_d, t_d_name in ((levenshtein_scores, 'levenshtein'), (levenshtein_pos_scores, 'levenshtein_pos'), (levenshtein_n_scores, 'levenshtein_n'), (jaccard_scores, 'jaccard'), (apted_scores, 'apted')
			):
			k = '%s / %s' % (m_d_name, t_d_name)
			v = scipy.stats.spearmanr(m_d, t_d).correlation
			results[k] = v

	return results


#### Entrypoint
if __name__ == "__main__":

	p = argparse.ArgumentParser()
	p.add_argument('--restrict_dataset_size', default=16384, type=int)
	p.add_argument('--sample_size', default=524288, type=int)
	p.add_argument('base_file')
	p.add_argument('--with_embs', default=None, type=str)
	p.add_argument('--annot_files', default=None, type=str)
	p.add_argument('--baseline_support', default=30, type=int)
	p.add_argument('--output_file', default='output.json', type=str)

	args = p.parse_args()

	print('loading models')

	predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
	if args.with_embs:
		import gensim
		meanings_model = gensim.models.KeyedVectors.load_word2vec_format(args.with_embs)
		print('reading defs')

		with open(args.base_file) as istr:
			lines = map(str.strip, istr)
			lines = (l.split('\t') for l in lines)
			meanings, sentences = zip(*lines)
			sentences = [tuple(s.split()) for s in sentences]

		meanings_idx = [meanings_model[d] for d in meanings]

	else:
		import torchvision
		import spacy
		torch.set_grad_enabled(False)
		nlp = spacy.load('en_core_web_sm')
		resnet = torchvision.models.resnet152(pretrained=True)
		dataset = torchvision.datasets.CocoCaptions(args.base_file, args.annot_files, transform=torchvision.transforms.ToTensor())
		if torch.cuda.is_available():
			resnet = resnet.cuda()
			meanings_idx = [
				i # repeat for coindexation
				for img, defs in dataset
				for i in [resnet(img.cuda().unsqueeze(0)).view(-1).cpu().numpy()] * len(defs)
			]
			resnet = resnet.cpu()
		else:
			meanings_idx = [
				i # repeat for coindexation
				for img, defs in dataset
				for i in [resnet(img.unsqueeze(0)).view(-1).numpy()] * len(defs)
			]
		sentences = [tuple([str(t) for t in nlp(d)]) for _, defs in dataset for d in defs]

	print('sampling')

	sample = sample_pairs(sentences, sample_size=args.sample_size, restrict_dataset_size=args.restrict_dataset_size)


	tree_idx = {}
	pos_decored_idx = {}
	for p in sample:
		for i in p:
			if not i in pos_decored_idx:
				parse = predictor.predict(sentence=' '.join(sentences[i]))
				tree_idx[i] = to_brkt(parse['trees'])
				pos_decored_idx[i] = sentences[i] + tuple(parse['pos_tags'])

	w2c = collections.defaultdict(itertools.count().__next__)

	print('computing correlations')

	true_score_results = correlation_fn([sentences, tree_idx, pos_decored_idx, meanings_idx, w2c, False, 'scores.json'])
	print(json.dumps(true_score_results))

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	single_arg = [sentences, tree_idx, pos_decored_idx, meanings_idx, w2c, True, False]

	baseline_results = list(pool.map(correlation_fn, itertools.repeat(single_arg, args.baseline_support)))
	json_output = {'true_score_results':true_score_results, 'baseline_results':baseline_results}
	print(json.dumps(json_output))
	with open(args.output_file, "w") as ostr:
		json.dump(json_output, ostr)
