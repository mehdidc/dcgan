from lightjob.cli import load_db
import sys
import os
sys.path.append('/home/mcherti/work/code/feature_generation')
from tools.common import to_generation, preprocess_gen_data
from tools.viz.obox import construct_image
import numpy as np
import click
import joblib
from keras.models import model_from_json
from tools.common import disp_grid
from skimage.io import imsave
from skimage.transform import resize
fonts = np.load('/home/mcherti/work/data/fonts/fonts.npz')
fonts_X = fonts['X']
fonts_y = fonts['y']
fonts_examples = [1 - fonts_X[fonts_y==c][0] / 255. for c in range(26)]
fonts_examples = [resize(x[0], (28, 28), preserve_range=True)[None, :, :] for x in fonts_examples]

def get_model(model_folder):
    arch = (open(os.path.join(model_folder, 'model.json'))).read()
    model = model_from_json(arch)
    model.load_weights(os.path.join(model_folder, 'model.pkl'))
    return model

@click.command()
@click.option('--field', default='stats.out_of_the_box_classification.fonts.objectness', required=False)
@click.option('--model-folder', default='tools/models/external/fonts', required=False)
@click.option('--name', default='', required=False)
def main(field, model_folder, name):
    model_folder = '/home/mcherti/work/code/feature_generation/' + model_folder
    db = load_db()
    jobs = db.jobs_with(state='success')
    indices = np.arange(len(jobs))
    objectness = map(lambda j:db.get_value(j, field, if_not_found=np.nan), jobs)
    objectness = np.array(objectness)
    indices = filter(lambda ind:not np.isnan(objectness[ind]), indices)
    indices = sorted(indices, key=lambda i:objectness[i])
    indices = indices[::-1]
    indices = indices[0:10]
    m =  0
    model = get_model(model_folder)
    for ind in indices:
        j = jobs[ind]
        print(j['summary'])
        filename = 'jobs/{}/images.npz'.format(j['summary'])
        X = joblib.load(filename)
        X = preprocess_gen_data(X)
        batch_size = 2048
        preds = []
        for i in range(0, len(X), batch_size):
            x = X[i:i+batch_size]
            preds.append(model.predict(x))
        preds = np.concatenate(preds, axis=0)
        if preds.shape[1] == 36:
            preds = preds[:, 10:]
        nb_classes = preds.shape[1]
        img = construct_image(X, preds, nbrows=6, nbcols=6, border=1, space=10, size=4, nb_classes=nb_classes)
        if not name:
            name = field + '_' + model_folder.split('/')[-1]
        folder = 'exported_data/obox/{}'.format(name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        print(folder)
        imsave('exported_data/obox/{}/{:05d}.png'.format(name, m + 1), img)
        m += 1

if __name__ == '__main__':
    main()
