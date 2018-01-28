import argparse

import numpy as np

from model import CNNModel


def get_data(run_time_arguments):
    data = np.load(run_time_arguments.data_filename[0])
    if run_time_arguments.reshape:
        data = data.reshape(run_time_arguments.reshape)
    print("data shape: ", data.shape)

    ss_offset = 22
    ss_labels8 = ['C', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']
    ss_labels3 = {'H': ['G', 'H', 'I'], 'E': ['E', 'B'], 'C': ['S', 'T', 'C'], 'NoSeq': ['NoSeq']}

    ss_column_indices = []
    for label3 in ['H', 'E', 'C', 'NoSeq']:
        ss_column_indices.append([ss_offset + ss_labels8.index(ss) for ss in ss_labels3[label3]])

    ss_columns = np.zeros([data.shape[0], data.shape[1], len(ss_labels3)])
    for i, column_indices in enumerate(ss_column_indices):
        ss_columns[:, :, i] = np.sum(data[:, :, column_indices], axis=2)

    data = np.concatenate((data[:, :, :ss_offset], data[:, :, 35:]), axis=2), ss_columns
    print("ss-reduced data set shape: ", data[0].shape, data[1].shape)
    training_data = data[0][slice(*run_time_arguments.training_range)], data[1][slice(*run_time_arguments.training_range)]
    print("training data shape: ", training_data[0].shape, training_data[1].shape)

    test_data = data[0][slice(*run_time_arguments.testing_range)], data[1][slice(*run_time_arguments.testing_range)]
    print("test data shape: ", test_data[0].shape, test_data[1].shape)

    valid_data = data[0][slice(*run_time_arguments.validating_range)], data[1][slice(*run_time_arguments.validating_range)]
    print("test data shape: ", valid_data[0].shape, valid_data[1].shape)

    # Training data and test data is available by
    train_x, train_y = training_data
    test_x, test_y = test_data
    valid_x, valid_y = valid_data

    # train_x = np.reshape(train_x, newshape=[np.shape(train_x)[0], np.shape(train_x)[1], np.shape(train_x)[2], 1])
    # test_x = np.reshape(test_x, newshape=[np.shape(test_x)[0], np.shape(test_x)[1], np.shape(test_x)[2], 1])
    # valid_x = np.reshape(valid_x, newshape=[np.shape(valid_x)[0], np.shape(valid_x)[1], np.shape(valid_x)[2], 1])
    return train_x, train_y, test_x, test_y, valid_x, valid_y


def get_end_of_peptide_idx(labels):
    end_of_peptide_idx = np.zeros(shape=[len(labels), 1], dtype=np.int32)
    for i in range(len(labels)):
        for j in range(len(labels[i, :])):
            if labels[i, j, 3] == 1.0:
                end_of_peptide_idx[i] = int(j)
                break
    return end_of_peptide_idx


def main(arguments):
    x_train, y_train, x_test, y_test, x_valid, y_valid = get_data(arguments)

    train_end_idx = get_end_of_peptide_idx(y_train)
    test_end_idx = get_end_of_peptide_idx(y_test)
    valid_end_idx = get_end_of_peptide_idx(y_valid)

    print("Here!")
    model = CNNModel()
    model.train(x_train, y_train, x_test, y_test, x_valid, y_valid, train_end_idx, test_end_idx, valid_end_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_filename", nargs=1, help="data file", metavar="FILE")
    parser.add_argument("--reshape", dest="reshape", nargs=3, default=(6133, -1, 57), help="Optionally rehape dataset",
                        type=int)
    parser.add_argument("--training-range", dest="training_range", nargs=2, default=(0, 5000),
                        help="Which rows to use for training", type=int)
    parser.add_argument("--testing-range", dest="testing_range", nargs=2, default=(5000, 5600),
                        help="Which rows to use for training", type=int)
    parser.add_argument("--validating-range", dest="validating_range", nargs=2, default=(5600, 6133),
                        help="Which rows to use for validation", type=int)
    args = parser.parse_args()
    main(args)
