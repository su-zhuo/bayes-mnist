import numpy as np
import readMNIST
from sklearn.decomposition import PCA
import warnings
import time

warnings.filterwarnings('ignore')


def Train(train_data, train_label):
    label = np.unique(train_label)
    num = train_data.shape[0]
    num_feature = train_data.shape[1]
    numlabel = label.shape[0]

    theta = np.zeros((numlabel, num_feature))
    sigma = np.zeros((numlabel, num_feature, num_feature))
    pyj = np.zeros(numlabel)

    for i, y in enumerate(label):
        # theta[i] = np.mean(train_data[np.squeeze(np.argwhere(train_label == y))[:, 0], :], axis=0)
        theta[i] = train_data[np.squeeze(np.argwhere(train_label == y))[:, 0], :].sum(0)/train_data.shape[0]
        sigma[i] = my_cov(train_data[np.squeeze(np.argwhere(train_label == y))[:, 0], :])
        # sigma[i] = my_var(train_data[np.squeeze(np.argwhere(train_label == y))[:, 0], :])
        pyj[i] = np.sum(train_label == y) / num  # prior
        # print(pyj[i])
    return theta, sigma, pyj

def my_cov(matrix):
    assert isinstance(matrix, np.ndarray)
    d = matrix.shape
    matrix = matrix - np.repeat((matrix.sum(0)/d[0]).reshape(-1, d[1]), d[0], axis=0)
    cov_ = np.dot(matrix.T, matrix)/(matrix.shape[0])
    return cov_

def my_var(matrix):
    assert isinstance(matrix, np.ndarray)
    d = matrix.shape
    _sum = np.sum(np.square(matrix - np.repeat((matrix.sum(0)/d[0]).reshape(-1, d[1]), d[0], axis=0)), axis=0)/d[0]
    cov_ = np.dot(matrix.T, matrix) / (matrix.shape[0])
    return cov_


def Test(test_data, test_label, theta, sigma, pyj):
    label = np.unique(test_label)
    num = test_data.shape[0]
    num_feature = test_data.shape[1]
    numlabel = label.shape[0]

    eps = np.eye(num_feature) * 0.01
    sigma_inv = np.zeros((numlabel, num_feature, num_feature))
    sigma_det = np.zeros(numlabel)
    for i in range(numlabel):
        # sigma[i] = sigma[i] + eps
        sigma_inv[i] = np.linalg.inv(sigma[i])
        sigma_det[i] = np.linalg.det(sigma[i])

    acc = 0

    for i in range(num):
        p_yj = np.zeros((numlabel))
        for j in range(numlabel):
            p_yj[j] = -np.dot(np.dot((test_data[i, :] - theta[j]).T, sigma_inv[j]), (test_data[i, :] - theta[j])) - np. \
                log(abs(sigma_det[j])) + np.log(pyj[j])
        p_yj_xi = np.argmax(p_yj)
        acc = acc + (p_yj_xi == test_label[i])
    print('test accuracy is :', acc / num)


import numpy as np


def pca(data_mat, top_n_feat=99999999):

    num_data, dim = data_mat.shape

    mean_vals = data_mat.sum(0)/data_mat.shape[0]
    mean_removed = data_mat - mean_vals

    cov_mat = my_cov(mean_removed)

    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
    eig_val_index = np.argsort(eig_vals)
    eig_val_index = eig_val_index[:-(top_n_feat + 1): -1]
    reg_eig_vects = eig_vects[:, eig_val_index]

    low_d_data_mat = mean_removed * reg_eig_vects
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals

    return low_d_data_mat, reg_eig_vects


if __name__ == "__main__":
    train_imgs = readMNIST.loadImageSet('train')
    train_labels = readMNIST.loadLabelSet('train')
    test_imgs = readMNIST.loadImageSet('test')
    test_labels = readMNIST.loadLabelSet('test')

    for i in range(15):
        dim = (i+1) * 5

        # print(test_imgs.shape)
        train_imgs_new, reg_eig_vects = pca(train_imgs, dim)
        train_imgs_new = train_imgs_new.A

        test_imgs_new = test_imgs * reg_eig_vects
        test_imgs_new = test_imgs_new.A
        # print(test_imgs_new.shape)

        # debug for PCA
        # n_components = 30
        # pca = PCA(n_components=n_components).fit(train_imgs)
        # train_imgs = pca.transform(train_imgs)
        # test_imgs = pca.transform(test_imgs)

        # print('train started')
        time_1 = time.time()
        theta, sigma, pyj = Train(train_imgs_new, train_labels)
        # print('test started')
        print("The dim is", dim)
        time_2 = time.time()
        Test(test_imgs_new, test_labels, theta, sigma, pyj)
        time_3 = time.time()

        print("The time is", time_2 - time_1)
