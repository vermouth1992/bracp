import torch
import torch.utils.data as data


class UniformBatchSampler(data.Sampler):
    def __init__(self, data_source, batch_size):
        super(UniformBatchSampler, self).__init__(data_source=data_source)
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            # update n in case data source is changing.
            n = len(self.data_source)
            yield torch.randint(high=n, size=torch.Size([self.batch_size]))
