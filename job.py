import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import joblib
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.metrics import nnc_score, nnd_score

import re
import shutil
import random
import numpy as np

from datetime import datetime

import sys
sys.path.append('/home/mcherti/work/code/feature_generation')
from jobs.hp import get_scores_bandit, get_hypers

def load_data():
    from datakit.mnist import load
    data = load(which='train')
    X, y = data['train']['X'], data['train']['y']
    X =  X / 255.
    nb_classes = 10
    return X, y, nb_classes

def sample():
    #np.random.seed(42)
    #random.seed(42)
    R = random
    rng = np.random
    hp = dict(
        k = R.choice((1, 2, 3)),             # # of discrim updates for each gen update
        l2 = R.choice((0, R.choice(np.logspace(-6, -1, 100))       )),       # l2 weight decay
        #b1 = R.choice(np.linspace(0, 1, 100)),          # momentum term of adam
        nbatch = 128,      # # of examples in batch
        nz = R.choice((10, 20, 50, 70, 100, 150, 200, 300)),          # # of dim for Z
        ngfc = R.choice((8, 16, 32, 64, 128, 256, 1024, 2048)),       # # of gen units for fully connected layers
        ndfc = R.choice((8, 16, 32, 64, 128, 256, 1024, 2048)),       # # of discrim units for fully connected layers
        ngf = R.choice((8, 16, 32, 64, 128, 256, 512)),        # # of gen filters in first conv layer
        ndf = R.choice((8, 16, 32, 64, 128, 256, 512)),          # # of discrim filters in first conv layer
        niter = R.choice((50, 100, 150, 200, 250, 300)),       # # of iter at starting learning rate
        niter_decay = 100, # # of iter to linearly decay learning rate to zero
        lr = R.choice((0.0002,  R.choice(np.logspace(-6, -1, 100))  )),       # initial learning rate for adam
        scale = R.choice(np.logspace(-3, -1, 100)),
        budget_hours=2
    )
    return hp

def run(hp, folder):
    trX, trY, nb_classes = load_data()
    k = 1             # # of discrim updates for each gen update
    l2 = 2.5e-5       # l2 weight decay
    b1 = 0.5          # momentum term of adam
    nc = 1            # # of channels in image
    ny = nb_classes   # # of classes
    nbatch = 128      # # of examples in batch
    npx = 28          # # of pixels width/height of images
    nz = 100          # # of dim for Z
    ngfc = 512       # # of gen units for fully connected layers
    ndfc = 512      # # of discrim units for fully connected layers
    ngf = 64          # # of gen filters in first conv layer
    ndf = 64          # # of discrim filters in first conv layer
    nx = npx*npx*nc   # # of dimensions in X
    niter = 200       # # of iter at starting learning rate
    niter_decay = 100 # # of iter to linearly decay learning rate to zero
    lr = 0.0002       # initial learning rate for adam
    scale = 0.02

    k = hp['k']
    l2 = hp['l2']
    #b1 = hp['b1']
    nc = 1
    ny = nb_classes
    nbatch = hp['nbatch']
    npx = 28
    nz = hp['nz']
    ngfc = hp['ngfc']       # # of gen units for fully connected layers
    ndfc = hp['ndfc']      # # of discrim units for fully connected layers
    ngf = hp['ngf']          # # of gen filters in first conv layer
    ndf = hp['ndf']          # # of discrim filters in first conv layer
    nx = npx*npx*nc   # # of dimensions in X
    niter = hp['niter']       # # of iter at starting learning rate
    niter_decay = hp['niter_decay'] # # of iter to linearly decay learning rate to zero
    lr = hp['lr']       # initial learning rate for adam


    scale = hp['scale']

    #k = 1             # # of discrim updates for each gen update
    #l2 = 2.5e-5       # l2 weight decay
    b1 = 0.5          # momentum term of adam
    #nc = 1            # # of channels in image
    #ny = nb_classes   # # of classes
    budget_hours = hp.get('budget_hours', 2)
    budget_secs = budget_hours * 3600

    ntrain = len(trX)
    def transform(X):
        return (floatX(X)).reshape(-1, nc, npx, npx)

    def inverse_transform(X):
        X = X.reshape(-1, npx, npx)
        return X
    
    model_dir = folder
    samples_dir = os.path.join(model_dir, 'samples')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    relu = activations.Rectify()
    sigmoid = activations.Sigmoid()
    lrelu = activations.LeakyRectify()
    bce = T.nnet.binary_crossentropy

    gifn = inits.Normal(scale=scale)
    difn = inits.Normal(scale=scale)

    gw  = gifn((nz, ngfc), 'gw')
    gw2 = gifn((ngfc, ngf*2*7*7), 'gw2')
    gw3 = gifn((ngf*2, ngf, 5, 5), 'gw3')
    gwx = gifn((ngf, nc, 5, 5), 'gwx')

    dw  = difn((ndf, nc, 5, 5), 'dw')
    dw2 = difn((ndf*2, ndf, 5, 5), 'dw2')
    dw3 = difn((ndf*2*7*7, ndfc), 'dw3')
    dwy = difn((ndfc, 1), 'dwy')

    gen_params = [gw, gw2, gw3, gwx]
    discrim_params = [dw, dw2, dw3, dwy]

    def gen(Z, w, w2, w3, wx, use_batchnorm=True):
        if use_batchnorm:
            batchnorm_ = batchnorm
        else:
            batchnorm_ = lambda x:x
        h = relu(batchnorm_(T.dot(Z, w)))
        h2 = relu(batchnorm_(T.dot(h, w2)))
        h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
        h3 = relu(batchnorm_(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
        x = sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
        return x

    def discrim(X, w, w2, w3, wy):
        h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
        h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
        h2 = T.flatten(h2, 2)
        h3 = lrelu(batchnorm(T.dot(h2, w3)))
        y = sigmoid(T.dot(h3, wy))
        return y

    X = T.tensor4()
    Z = T.matrix()

    gX = gen(Z, *gen_params)

    p_real = discrim(X, *discrim_params)
    p_gen = discrim(gX, *discrim_params)

    d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
    d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
    g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

    d_cost = d_cost_real + d_cost_gen
    g_cost = g_cost_d

    cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

    lrt = sharedX(lr)
    d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
    g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
    d_updates = d_updater(discrim_params, d_cost)
    g_updates = g_updater(gen_params, g_cost)
    #updates = d_updates + g_updates

    print 'COMPILING'
    t = time()
    _train_g = theano.function([X, Z], cost, updates=g_updates)
    _train_d = theano.function([X, Z], cost, updates=d_updates)
    _gen = theano.function([Z], gX)
    print '%.2f seconds to compile theano functions'%(time()-t)

    tr_idxs = np.arange(len(trX))
    sample_zmb = floatX(np_rng.uniform(-1., 1., size=(200, nz)))

    def gen_samples(n, nbatch=128):
        samples = []
        labels = []
        n_gen = 0
        for i in range(n/nbatch):
            zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
            xmb = _gen(zmb)
            samples.append(xmb)
            n_gen += len(xmb)
        n_left = n-n_gen
        zmb = floatX(np_rng.uniform(-1., 1., size=(n_left, nz)))
        xmb = _gen(zmb)
        samples.append(xmb)
        return np.concatenate(samples, axis=0)

    s = floatX(np_rng.uniform(-1., 1., size=(10000, nz)))
    n_updates = 0
    n_check = 0
    n_epochs = 0
    n_updates = 0
    n_examples = 0
    t = time()
    begin = datetime.now()
    for epoch in range(1, niter+niter_decay+1): 
        t = time()
        print("Epoch {}".format(epoch))
        trX = shuffle(trX)
        for imb in tqdm(iter_data(trX, size=nbatch), total=ntrain/nbatch):
            imb = transform(imb)
            zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
            if n_updates % (k+1) == 0:
                cost = _train_g(imb, zmb)
            else:
                cost = _train_d(imb, zmb)
            n_updates += 1
            n_examples += len(imb)
        samples = np.asarray(_gen(sample_zmb))
        grayscale_grid_vis(inverse_transform(samples), (10, 20), '{}/{:05d}.png'.format(samples_dir, n_epochs))
        n_epochs += 1
        if n_epochs > niter:
            lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
        if n_epochs % 50 == 0 or epoch == niter + niter_decay or epoch == 1:
            imgs = []
            for i in range(0, s.shape[0], nbatch):
                imgs.append(_gen(s[i:i+nbatch]))
            img = np.concatenate(imgs, axis=0)
            samples_filename = '{}/{:05d}_gen.npz'.format(model_dir, n_epochs)
            joblib.dump(img, samples_filename, compress=9)
            shutil.copy(samples_filename, '{}/gen.npz'.format(model_dir))
            joblib.dump([p.get_value() for p in gen_params], '{}/d_gen_params.jl'.format(model_dir, n_epochs), compress=9)
            joblib.dump([p.get_value() for p in discrim_params], '{}/discrim_params.jl'.format(model_dir, n_epochs), compress=9)
        print('Elapsed : {}sec'.format(time() - t))

        if (datetime.now() - begin).total_seconds() >= budget_secs:
            print("Budget finished.quit.")
            break


def insert(nb=1):
    from lightjob.cli import load_db
    from lightjob.utils import summarize
    db = load_db()
    nb_samples = 100
    target = 'stats.out_of_the_box_classification.fonts.objectness'
    inputs, outputs = get_hypers(state='success', y_col=target)
    for _ in range(nb):
        new_inputs = [sample() for _ in range(nb_samples)]
        scores = get_scores_bandit(inputs, outputs, new_inputs=new_inputs, algo='ucb')
        new_input = new_inputs[np.argmax(scores)]
        if db.job_exists_by_summary(summarize(new_input)):
            existing = '(exists)'
        else:
            existing = '(new)'
        print('expected {} for the selected job : {}, id:{}{}'.format(target, np.max(scores), summarize(new_input), existing))
        db.safe_add_job(new_input)

def run_job(s):
    from lightjob.cli import load_db
    db = load_db()
    j = db.get_job_by_summary(s)
    db.modify_state_of(j['summary'], 'running')
    try:
        run(j['content'], 'jobs/{}'.format(j['summary']))
    except Exception as ex:
        print('error : {}'.format(ex))
        db.modify_state_of(j['summary'], 'error')
    else:
        db.modify_state_of(j['summary'], 'success')
 
if __name__ == '__main__':
    run_job(sys.argv[1])
