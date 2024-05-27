import logging

from torch.utils.data.sampler import BatchSampler

from .dataset_utils import check_to_log

LOGGER = logging.getLogger(__name__)


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.sampler = self.batch_sampler.sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter
        if check_to_log():
            LOGGER.info(f"Build IterationBasedBatchSampler with {num_iterations} iterations. Start from {start_iter}th iteration.")

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations