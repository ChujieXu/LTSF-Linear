import numpy as np


def get_mse(benchmark_file, result_file):
    true = np.load(benchmark_file)
    pred = np.load(result_file)
    return np.mean((pred - true) ** 2)


if __name__ == "__main__":
    benchmark_file = "dataset/test/data_value_test_y.npy"
    result_file = "results/data_value_test_y.npy"
    metric = 7
    result = get_mse(benchmark_file, result_file)
    print(result)
