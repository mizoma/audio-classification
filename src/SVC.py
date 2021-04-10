from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def set_pipeline(dim_reduction_fn):

    dim_method = dim_reduction_fn
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(dim_method, svc)

    return model


def perform_grid_search(model, Xtrain, ytrain, Xtest, param_grid=None):

    if param_grid is None:
        param_grid = {
            'svc__C': [1, 5, 10, 50],
            'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]
            }

    grid = GridSearchCV(model, param_grid)
    grid.fit(Xtrain, ytrain)
    print(grid.best_estimator_)

    model = grid.best_estimator_
    yfit = model.predict(Xtest)

    return yfit

def model_evaluation(Xtest, ytest, yfit, audio):

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 8, figsize=(20,10))
    for i, axi in enumerate(ax.flat):
        axi.imshow(Xtest[i].reshape(128, 128), cmap='bone')
        axi.set(xticks=[], yticks=[])
        axi.set_xlabel(audio.target_names_single[int(yfit[i])].split()[-1],
                    color='black' if yfit[i] == ytest[i] else 'red')
    fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)

    print(classification_report(ytest, yfit))