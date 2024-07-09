from trainer_lib import datasets
import numpy as np

if __name__ == '__main__':
    sample = np.random.rand(4*2+1+4, 1)
    config = datasets.TimeSeriesWindowedDatasetConfig(2, 1, 4, 1, 4, True)
    data = datasets.TimeSeriesWindowedTensorDataset(sample, config)
    print('sample: ', sample)
    print('data:', data[1])
    print('len data:', len(data))
