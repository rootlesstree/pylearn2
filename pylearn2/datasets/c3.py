# We'll need numpy to manage arrays of data
import numpy as np
from os import listdir
from os.path import join
# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import cv2
 
def load_data():
    """
    Loads the red wine quality dataset from:
 
    P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
    Modeling wine preferences by data mining from physicochemical properties.
    In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
 
    The dataset contains 1,599 examples, including a floating point regression
    target.
 
    Parameters
    ----------
    start: int
    stop: int
 
    Returns
    -------
 
    dataset : DenseDesignMatrix
        A dataset include examples start (inclusive) through stop (exclusive).
        The start and stop parameters are useful for splitting the data into
        train, validation, and test data.
    """
    
    label_class = {"melt":0,"plus_metal":1,"shadowing":2}
    path = "/home/rr/PYLEARN_DATA/C3_sy/"

    train_path = path + "train/"
    test_path = path + "test/"
    
    train_matrix = []

    
    for defect in listdir(train_path):
        temp_x = []
        temp_label = []
        for files in listdir(join(train_path,defect)):
            temp_x.append(np.reshape(cv2.resize( cv2.equalizeHist(cv2.imread(join(train_path,defect,files),0)),(100,100)),10000))
            temp_label.append(label_class[defect])
        temp_x = np.asarray(temp_x)/float(255)
        temp_label = np.asarray(temp_label)
        temp_label = temp_label.reshape(temp_label.shape[0],1)
        train_matrix.append(DenseDesignMatrix(X=temp_x,y=temp_label))


    test_matrix = []


    for defect in listdir(test_path):
        temp_x = []
        temp_label = []



        for files in listdir(join(test_path,defect)):
            temp_x.append(np.reshape(cv2.resize(cv2.equalizeHist(cv2.imread(join(test_path,defect,files),0)),(100,100)),10000))
            temp_label.append(label_class[defect])
        temp_x = np.asarray(temp_x)/float(255)
        temp_label = np.asarray(temp_label)
        temp_label = temp_label.reshape(temp_label.shape[0],1)
        test_matrix.append(DenseDesignMatrix(X=temp_x,y=temp_label))




    return train_matrix,test_matrix




if __name__ == '__main__':
    load_data()
