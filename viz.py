import os
from lightjob.cli import load_db
from tqdm import tqdm
from skimage.io import imread, imsave
import json
from collections import OrderedDict
import argparse
import joblib
import numpy as np
import sys
sys.path.append('/home/mcherti/work/code/feature_generation')
from tools.common import disp_grid, to_training
from helpers import mkdir_path
from joblib import Parallel, delayed
import click

def generate_one(j, per_jobset=True):
    db = load_db()
    id_ = j['summary']
    img_filename = 'jobs/results/{}/images.npz'.format(id_)
    data = joblib.load(img_filename)
    data = np.array(data)
    if len(data.shape) == 5:
        # (10000, 101, 1, 28, 28)
        data = data[:, -1]
    elif len(data.shape) == 3:
        # (10000, 28, 28)
        data = data[:, None]
    if len(data) == 0:
        return
    data = np.clip(data, 0, 1)
    data = data[0:1000]
    img = disp_grid(data, border=1, bordercolor=(0.3, 0, .0), normalize=False)
    imsave('exported_data/figs/generated/{}.png'.format(id_), img)
    print('completed {}'.format(id_))

@click.command()
@click.option('--where', default=None)
@click.option('--per-jobset/--no-per-jobset', default=True)
def generate(where, per_jobset):
    kw = {}
    db = load_db()
    J = db.jobs_with(state='success')
    if where:
        ref_jobs = set(map(lambda j:j['summary'], db.jobs_with(where=where)))
        J = filter(lambda j:j['content']['model_summary'] in ref_jobs, J)
    print('Nb of jobs : {}'.format(len(J)))
    Parallel(n_jobs=1, verbose=1)(delayed(generate_one)(j, per_jobset=per_jobset) for j in J)

if __name__ == '__main__':
    generate()
