import os
import joblib
from joblib import Parallel, delayed
from datakit import mnist
from likelihood_estimation_parzen import ll_parzen
import numpy as np
import sys
sys.path.append('/home/mcherti/work/code/feature_generation')
from tools.compute.genstats import compute_out_of_the_box_classification, out_of_the_box_names, out_of_the_box_models
import pprint
data = mnist.load(which='test')
mnist_test_X = data['test']['X'] / 255.
mnist_test_X = mnist_test_X.astype(np.float32)

def flatten(x):
    return x.reshape((x.shape[0], -1))

def compute_stats(j):
    print('Starting {}...'.format(j['summary']))
    st = {}
    stat_funcs = [parzen, out_of_the_box]
    stats = j.get('stats', {})
    if not stats:
        stats = {}
    for func in stat_funcs:
        st = func(j)
        stats.update(st)
    print('Finished {}'.format(j['summary']))
    #pprint.pprint(stats, indent=2)
    return stats

def has(j, stat):
    if 'stats' in j and j['stats'] and stat in j['stats']:
        return True
    else:
        return False

def parzen(j):
    if has(j, 'll_parzen_mean'):
        return {}
    samples = joblib.load(os.path.join('jobs', j['summary'], 'gen.npz'))
    samples = samples[0:10000]
    sigma = 0.23
    ll = ll_parzen(sigma=sigma, samples=flatten(samples), test_X=flatten(mnist_test_X), batch_size=10)
    ll = np.array(ll)
    return {'ll_parzen_mean': float(ll.mean()), 'll_parzen_std': float(ll.std() / (100.)) }

def out_of_the_box(j):
    #if has(j, 'out_of_the_box_classification'):
    #    return {}
    folder = os.path.join('jobs', j['summary'])
    try:
        os.link(folder + '/gen.npz', folder + '/images.npz')
    except OSError:
        pass
    names = out_of_the_box_names
    models = out_of_the_box_models
    models = ['../../../feature_generation/' + m for m in models]
    stat = {}
    for model_name, model_folder in zip(names, models):
        stat[model_name] = compute_out_of_the_box_classification(folder, model_name, model_folder, nb_samples=1000)
    return {'out_of_the_box_classification': stat}

if __name__ == '__main__':
    from lightjob.cli import load_db
    n_jobs = 1
    db = load_db()
    jobs = db.jobs_with(state='success')
    stats = Parallel(n_jobs=n_jobs)(delayed(compute_stats)(j) for j in jobs)
    for j, s in zip(jobs, stats):
        db.job_update(j["summary"], dict(stats=s))
