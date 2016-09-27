from sklearn.svm import SVC as classifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

#classifier pipeline
clf_pipeline = clf_pipeline = OneVsRestClassifier(
                Pipeline([('reduce_dim', RandomizedPCA()),
                          ('clf', classifier())
                          ]
                         ))

C_range = 10.0 ** np.arange(-2, 9)
gamma_range = 10.0 ** np.arange(-5, 4)
n_components_range = (10, 100, 200)
degree_range = (1, 2, 3, 4)

param_grid = dict(estimator__clf__gamma=gamma_range,
                  estimator__clf__c=c_range,
                  estimator__clf__degree=degree_range,
                  estimator__reduce_dim__n_components=n_components_range)

grid = GridSearchCV(clf_pipeline, param_grid,
                                cv=StratifiedKFold(y=Y, n_folds=3), n_jobs=1,
                                verbose=2)
grid.fit(X, Y)
>>> from sklearn.metrics import classification_report
>>> y_true = [0, 1, 2, 2, 2]
>>> y_pred = [0, 0, 2, 2, 1]
>>> target_names = ['class 0', 'class 1', 'class 2']
>>> print(classification_report(y_true, y_pred, target_names=target_names))
