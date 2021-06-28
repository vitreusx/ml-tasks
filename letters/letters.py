import numpy as np
import cv2
import sys
from copy import deepcopy
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from jinja2 import Template
from pathlib import Path

in_file = sys.argv[1]
paths = np.array([x.strip() for x in open(in_file, 'r').readlines()])
images = {path: cv2.imread(path) for path in paths}

w = np.array([im.shape[1] for im in images.values()])
h = np.array([im.shape[0] for im in images.values()])

punct_marks = h < 10
sequences = w > 15
chars = (h >= 10) & (w <= 13)

punct_clusters = [
    paths[(h < 10) & (w/h > 2)],
    paths[(h < 10) & (w/h < 2) & (h/w > 1.5)],
    paths[(h < 10) & (w/h < 2) & (h/w <= 1.5)],
]

def rescale(image):
    x = image
    x = cv2.resize(x, (16, 16), interpolation=cv2.INTER_CUBIC)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    _, x = cv2.threshold(x, 0, 255, cv2.THRESH_OTSU)
    return x

rescaled = {path: rescale(images[path])
            for path in paths}

X = np.array([np.asarray(rescaled[path])
              for path in paths])
X = X / 255

X_sk = X.reshape((-1, 16*16))
X_sk = np.hstack([X_sk, w.reshape((-1, 1)), h.reshape((-1, 1))])
tX_sk = PCA(n_components=64).fit_transform(X_sk)

Xc_sk = X_sk[chars]
tXc_sk = PCA(n_components=64).fit_transform(Xc_sk)

char_clust = AgglomerativeClustering(n_clusters=32).fit(tXc_sk)
char_vals = np.unique(char_clust.labels_)

Xs_sk = X_sk[sequences]
tXs_sk = PCA(n_components=64).fit_transform(Xs_sk)

seq_clust = OPTICS(min_samples=25).fit(tXs_sk)
seq_vals = np.unique(seq_clust.labels_)

clusters = [
    *punct_clusters,
    *(paths[chars][char_clust.labels_ == v] for v in char_vals),
    *(paths[sequences][seq_clust.labels_ == v] for v in seq_vals if v >= 0),
    *([seq] for seq in paths[sequences][seq_clust.labels_ == -1])
]

html = Template(open('output.j2', 'r').read())
out_html = html.render(clusters=clusters)
open('output.html', 'w').write(out_html)

out_txt = '\n'.join(' '.join(Path(item).name for item in cluster)
                    for cluster in clusters)
open('output.txt', 'w').write(out_txt)