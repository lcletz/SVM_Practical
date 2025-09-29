#%%
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
# from sklearn.utils import shuffle # train_test_split used
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
#%%
###############################################################################
#               Toy dataset : 2 gaussians
###############################################################################
random.seed(9204)

n1 = 200
n2 = 200
mu1 = [1., 1.]
mu2 = [-1./2, -1./2]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

plt.show()
plt.close("all")
plt.ion()
plt.figure(1, figsize=(15, 5))
plt.title('Premier jeu de données')
plot_2d(X1, y1)

#%%

X_train = X1[::2]
Y_train = y1[::2].astype(int)
X_test = X1[1::2]
Y_test = y1[1::2].astype(int)

# fit the model with linear kernel
clf = SVC(kernel='linear')
clf.fit(X_train, Y_train)

# predict labels for the test data base
y_pred = clf.predict(X_test)

# check your score
score = clf.score(X_test, Y_test)
print('Paramètre de régularisation C :', clf.C)
print('Paramètre de fonction noyau :', clf.kernel)
print('Score : %s' % score)

# display the frontiere
def f(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f, X_train, Y_train, w=None, step=50, alpha_choice=1)

#%%
# Same procedure but with a grid search
parameters = {'kernel': ['linear'], 'C': list(np.linspace(0.001, 3, 21))}
clf2 = SVC()
clf_grid = GridSearchCV(clf2, parameters, n_jobs=-1)
clf_grid.fit(X_train, Y_train)

# check your score
print('Meilleur paramètre de régularisation C :', clf_grid.best_estimator_.C)
print('Paramètre de fonction noyau choisi :', clf_grid.best_estimator_.kernel)
print('Score : %s' % clf_grid.score(X_test, Y_test))

def f_grid(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_grid.predict(xx.reshape(1, -1))

# display the frontiere
plt.figure()
frontiere(f_grid, X_train, Y_train, w=None, step=50, alpha_choice=1)

#%%
###############################################################################
#               Iris Dataset
###############################################################################
random.seed(9204)

iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

# split train test (say 25% for the test)
# You can shuffle and then separate or you can just use train_test_split 
#whithout shuffling (in that case fix the random state (say to 42) for reproductibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#%%
###############################################################################
# fit the model with linear vs polynomial kernel
###############################################################################
# Q1 Linear kernel
random.seed(9204)

# fit the model and select the best hyperparameter C
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}
svm1 = svm.SVC()
clf_linear = GridSearchCV(svm1, parameters, n_jobs=-1)
random.seed(9204)
clf_linear.fit(X_train, y_train)

# compute the score
print(clf_linear.best_params_)
print('Score de généralisation pour le noyau linéaire : %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))
#%%
# Q2 polynomial kernel
random.seed(9204)

Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[2, 3] # degree = 1 removed because it is equivalent to linear kernel and is the optimal degree for this dataset

# fit the model and select the best set of hyperparameters
parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
svm2 = svm.SVC()
clf_poly = GridSearchCV(svm2, parameters, n_jobs=-1)
clf_poly.fit(X_train, y_train)

print('Score de généralisation pour le noyau polynomial : %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))

#%%
# display your results using frontiere (svm_source.py)
random.seed(9204)

def f_linear(xx):
    return clf_linear.predict(xx.reshape(1, -1)) # reshaping to avoid warning

def f_poly(xx):
    return clf_poly.predict(xx.reshape(1, -1))

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title('Jeu de données "iris"')

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("Noyau linéaire")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("Noyau polynomial")
plt.tight_layout()
plt.draw()

#%%
###############################################################################
#               SVM GUI
###############################################################################

# please open a terminal and run python svm_gui.py
# Then, play with the applet : generate various datasets and observe the
# different classifiers you can obtain by varying the kernel

# https://scikit-learn.org/1.2/auto_examples/applications/svm_gui.html 

#%%
###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
random.seed(9204)

images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()
target_names

#%%
####################################################################
random.seed(9204)

# Pick a pair to classify such as
# names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']
names = ['George W Bush', 'Hugo Chavez']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))]
y = y.astype(int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

#%%
####################################################################
# Extract features
random.seed(9204)

for name in names:
    name_index = target_names.index(name)
    count = np.sum(lfw_people.target == name_index)
    print(f"{name} : {count} images")

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#%%
####################################################################
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)
random.seed(9204)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set

#%%
# Q4
random.seed(9204)

print("--- Noyau linéaire ---")
print('Ajustement du classifier aux données "train"')
t0 = time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
train_scores = []
test_scores = []
for C in Cs:
    parameters = {'kernel': ['linear'], 'C': [C]} # [C] to have a list of one element (TypeError otherwise)
    svm_model = svm.SVC()
    clf = GridSearchCV(svm_model, parameters, n_jobs=-1)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

ind_train = np.argmax(train_scores)
ind_test = np.argmax(test_scores)
print("Effectué en  %0.3fs" % (time() - t0))
print("Meilleur C (train): {}".format(Cs[ind_train]))
print("Meilleur C (test): {}".format(Cs[ind_test])) # done this to avoid overfitting I saw in the previous version

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(Cs, train_scores, color='#7B1FA2', marker='o', label='Score d\'entraînement')
ax.plot(Cs, test_scores, color='#4A148C', marker='s', label='Score de test')

ax.set_xlabel("Paramètres de régularisation C", fontsize=12, color='black')
ax.set_ylabel("Scores", fontsize=12, color='black')
ax.set_title("Courbes d'apprentissage : Train vs Test", fontsize=14, color='black', fontweight='normal')

ax.set_xscale("log")

ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors='black')
ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)

legend = ax.legend(framealpha=0.9, loc='best')
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('gray')

plt.tight_layout()
plt.show()
print("Meilleur train score: {:.4f}".format(np.max(train_scores)))
print("Meilleur test score: {:.4f}".format(np.max(test_scores)))
print("Écart sur-apprentissage au meilleur C (test): {:.4f}".format(train_scores[ind_test] - test_scores[ind_test]))

#%%
print('Prédiction du nom des personnes apparaissant dans les données "test"')
t0 = time()

# predict labels for the X_test images with the best classifier (basé sur le score de test)
parameters = {'kernel': ['linear'], 'C': [Cs[ind_test]]}
svm_model = svm.SVC()
clf = GridSearchCV(svm_model, parameters, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Effectué en  %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Niveau de chance : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Précision : %s" % clf.score(X_test, y_test))

#%%
####################################################################
# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# Look at the coefficients
plt.figure(figsize=(8, 6))
plt.imshow(np.reshape(clf.best_estimator_.coef_[0], (h, w)))
plt.colorbar(label='Poids des coefficients')
plt.title('Coefficients du SVM linéaire')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.show()


#%%
# Q5
random.seed(9204)

def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters, n_jobs=-1)
    _clf_linear.fit(_X_train, _y_train)

    print('Score de généralisation pour le noyau linéaire : %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
run_svm_cv(X, y)

print("Score avec variable de nuisance")
n_features = X.shape[1]
# On rajoute des variables de nuisances
sigma = 1
noise = sigma * np.random.randn(n_samples, 300)
#with gaussian coefficients of std sigma
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X_noisy.shape[0])] 
run_svm_cv(X_noisy, y)

#%%
# Q6
random.seed(9204)

print("Score après réduction de dimension")

# n_components = 20  # jouer avec ce parametre
# n_components = 10 # similar results as n_components = 5 but takes much longer
n_components = 5 
X_noisy_scaled = scaler.fit_transform(X_noisy)
pca = PCA(n_components=n_components, svd_solver='randomized').fit(X_noisy_scaled)
X_noisy_pca = pca.transform(X_noisy_scaled)

t0 = time()
run_svm_cv(X_noisy_pca, y)
print("Effectué en  %0.3fs" % (time() - t0))

#%%