import numpy as np
from sklearn.svm._base import BaseSVC
from sklearn.svm import SVC

class ImbalancedSVC(BaseSVC):
    def __init__(self, *, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, 
    shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None, 
    verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, 
    random_state=None):

        self._impl = 'c_svc'

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, nu=0., shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state)
    
    def fit(self, X, y, sample_weight=None):

        assert self.kernel == 'linear'

        clf = super().fit(X, y, sample_weight)

        assert len(self.classes_) == 2

        self.n_classes_ = np.zeros(clf.classes_.shape).astype(np.int16)
        self.n_datapoints_ = X.shape[0]
        for i, c in enumerate(clf.classes_):
            self.n_classes_[i] = np.count_nonzero(y == c)

        return self
        

    def predict(self, X):
        pos = np.cumsum(self.n_support_) - 1

        w_ = self.coef_[0][:-1]
        wd = self.coef_[0][-1]
        x_ = self.support_vectors_[pos[1]][:-1]
        xd = self.support_vectors_[pos[1]][-1]

        # bias of margin for positive class
        alpha = xd + (w_ @ x_)/wd

        x_ = self.support_vectors_[pos[0]][:-1]
        xd = self.support_vectors_[pos[0]][-1]

        # bias of margin of negative class
        beta = xd + (w_ @ x_)/wd

        # find new bias term 
        intercept_new_ = -wd*((alpha*self.n_classes_[0] + beta*self.n_classes_[1])/(self.n_datapoints_))
        self._intercept_ = np.array([intercept_new_])

        # intercept_new_ = -wd*(alpha + beta)/2

        return (super().predict(X))