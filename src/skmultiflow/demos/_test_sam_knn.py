import logging
import numpy as np
from skmultiflow.lazy import SAMKNN
from sklearn.metrics import accuracy_score
from skmultiflow.utils.utils import get_dimensions


def run(X, y, hyperParams):
    """ run
    
    Test function for SAMKNN, not integrated with evaluation modules.
    
    Parameters
    ----------
    X: numpy.ndarray of shape (n_samples, n_features)
        The feature's matrix, coded as 64 bits.
    
    y: numpy.array of size n_samples
        The labels for all the samples in X coded as 8 bits.
    
    hyperParams: dict
        A dictionary containing the __init__ params for the SAMKNN.
    
    """
    r, c = get_dimensions(X)
    classifier = SAMKNN(n_neighbors=hyperParams['nNeighbours'],
                        maxSize=hyperParams['maxSize'],
                        knnWeights=hyperParams['knnWeights'],
                        STMSizeAdaption=hyperParams['STMSizeAdaption'],
                        useLTM=hyperParams['useLTM'])
    logging.info('applying model on dataset')
    predicted_labels = []
    true_labels = []
    for i in range(r):
        pred = classifier.predict(np.asarray([X[i]]))
        predicted_labels.append(pred[0])
        true_labels.append(y[i])
        classifier = classifier.partial_fit(np.asarray([X[i]]), np.asarray([y[i]]), None)
        if (i % (r // 20)) == 0:
            logging.info(str((i // (r / 20))*5) + "%")
    accuracy = accuracy_score(true_labels, predicted_labels)
    logging.info('error rate %.2f%%' % (100-100*accuracy))


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    hyperParams ={'maxSize': 1000, 'nNeighbours': 5, 'knnWeights': 'distance', 'STMSizeAdaption': 'maxACCApprox', 'useLTM': False}
    # hyperParams = {'windowSize': 5000, 'nNeighbours': 5, 'knnWeights': 'distance', 'STMSizeAdaption': None,
    #               'useLTM': False}

    logging.info('loading dataset')
    # X=pd.read_csv('../datasets/neweather_data.csv', sep=',', header=None).values
    # y=pd.read_csv('../datasets/neweather_class.csv', sep=',', header=None, dtype=np.int8).values.ravel()
    X = np.loadtxt('../data/datasets/movingSquares.data')
    y = np.loadtxt('../data/datasets/movingSquares.labels', dtype=np.uint8)
    logging.info('%d samples' % X.shape[0])
    logging.info('%d dimensions' % X.shape[1])
    run(X[:], y[:], hyperParams)

