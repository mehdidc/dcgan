import os
import joblib
from lightjob.cli import load_db
import shutil

def rm(path):
    if not os.path.exists(path):
        return
    print('Removing {}'.format(path))
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

""" 
db = load_db()
jobs = db.jobs_with(state='success')
for j in jobs:
    folder = 'jobs/{}'.format(j['summary'])
    filenames = os.listdir(folder)
    filenames = filter(lambda f:'_gen.npz_01' in f, filenames)
    filenames = sorted(filenames, key=lambda f:int(f[0:5]))
    for f in filenames[0:-1]:
        rm('{}/{}'.format(folder, f))
    print(folder, filenames[0], filenames[0:-1], filenames[-1])
"""
"""
jobs = db.jobs_with(state='available')
S = set([j['summary'] for j in jobs])
for f in os.listdir('jobs'):
    print(f, f in S)
    f =  'jobs/{}'.format(f)
    if os.path.exists(f):
        pass
    else:
        print(f)
"""
