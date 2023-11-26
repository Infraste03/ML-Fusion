
# Run the model on test sets
# Output is prediction map of same shape as y_test


import numpy as np
import torch


def test_clf(model, X_test, y_test=None, sample_radius=8, batch_test=64):
    """
    The function `test_clf` takes a trained model and test data, and returns the predicted output for
    each test sample.
    
    :param model: The model parameter is the machine learning model that you want to test. It should be
    an instance of a trained model that has a predict method
    :param X_test: X_test is a 3-dimensional numpy array representing the test data. The dimensions are
    (number of samples, number of rows, number of columns)
    :param y_test: The `y_test` parameter is an optional input that represents the ground truth labels
    for the test samples. If `y_test` is provided, the function will only process test samples where the
    corresponding label in `y_test` is greater than 0. If `y_test` is not provided,
    :param sample_radius: The sample_radius parameter determines the size of the sample window around
    each pixel in the input image. It specifies the number of pixels in each direction (up, down, left,
    right) from the center pixel that will be included in the sample window, defaults to 8 (optional)
    :param batch_test: The parameter `batch_test` is the number of samples to process in each batch. It
    is used to control the memory usage during testing. The function will process `batch_test` number of
    samples at a time and then move on to the next batch until all samples have been processed, defaults
    to 64 (optional)
    :return: the output predictions for the test samples. If the `y_test` parameter is provided, it
    returns the predictions for all test samples. If `y_test` is not provided, it returns the
    predictions reshaped to match the shape of the input test data.
    """
    
    # input y_test if output test samples only
    data_rows = X_test.shape[1]
    data_cols = X_test.shape[2]

    model.eval()
    pred_all = []
    x_batch = []
    batch_count = 0
    for r in range(sample_radius, data_rows-sample_radius):
        for c in range(sample_radius, data_cols-sample_radius):
            if y_test is not None:
                if y_test[r-sample_radius, c-sample_radius] > 0:
                    x = X_test[:, r-sample_radius:r+sample_radius+1, c-sample_radius:c+sample_radius+1]
                    x_batch.append(x)
                    batch_count += 1
                    if batch_count == batch_test:
                        x_batch = torch.Tensor(np.array(x_batch))
                        x_batch = x_batch.cuda()
                        output = model(x_batch)
                        _, pred = torch.max(output.data, 1)
                        pred += 1
                        pred_all.append(pred.cpu().numpy())
                        batch_count = 0
                        x_batch = []
            else:
                x = X_test[:, r-sample_radius:r+sample_radius+1, c-sample_radius:c+sample_radius+1]
                x_batch.append(x)
                batch_count += 1
                if batch_count == batch_test:
                    x_batch = torch.Tensor(np.array(x_batch))
                    x_batch = x_batch.cuda()
                    output = model(x_batch)
                    _, pred = torch.max(output.data, 1)
                    pred += 1
                    pred_all.append(pred.cpu().numpy())
                    batch_count = 0
                    x_batch = []
                    
    if batch_count > 0:
        x_batch = torch.Tensor(np.array(x_batch))
        x_batch = x_batch.cuda()
        output = model(x_batch)
        _, pred = torch.max(output.data, 1)
        pred += 1
        pred_all.append(pred.cpu().numpy())
        
    output = np.concatenate(pred_all)

    if y_test is not None:
        return output
    else:
        return output.reshape(X_test.shape[1]-2*sample_radius, X_test.shape[2]-2*sample_radius)

