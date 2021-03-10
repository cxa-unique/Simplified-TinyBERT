# This code is for data reading in knowledge distillation

import logging
import torch
import numpy as np
import linecache
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)


class PregeneratedDataset(Dataset):
    def __init__(self, features_file, cache_dir, max_seq_length, num_examples, reduce_memory=True):
        logger.info('features_file: {}'.format(features_file))
        self.seq_len = max_seq_length
        self.num_samples = num_examples

        if reduce_memory:
            self.working_dir = Path(cache_dir)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(self.num_samples, self.seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
            label_ids = np.memmap(filename=self.working_dir/'label_ids.memmap',
                                  shape=(self.num_samples, ), mode='w+', dtype=np.int32)
            label_ids[:] = -1
        else:
            raise NotImplementedError

        logging.info("Loading training examples.")

        with open(features_file, 'r') as f:
            for i, line in enumerate(tqdm(f, total=self.num_samples, desc="Examples")):
                if i == 0:
                    continue
                tokens = line.strip().split(',')
                input_ids[i-1] = [int(id) for id in tokens[1].split()]
                input_masks[i-1] = [int(id) for id in tokens[2].split()]
                segment_ids[i-1] = [int(id) for id in tokens[3].split()]
                guid = tokens[0]
                label_ids[i-1] = int(tokens[4])

                if label_ids[i-1] != 0 and label_ids[i-1] != 1:
                    raise KeyError

                if i < 2:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % guid)
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids[i-1]]))
                    logger.info("input_masks: %s" % " ".join([str(x) for x in input_masks[i-1]]))
                    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids[i-1]]))
                    logger.info("label: %s" % str(label_ids[i-1]))

        logging.info("Loading complete!")
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.label_ids = label_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item], dtype=torch.long),
                torch.tensor(self.input_masks[item], dtype=torch.long),
                torch.tensor(self.segment_ids[item], dtype=torch.long),
                torch.tensor(self.label_ids[item], dtype=torch.long))


def distill_dataloader(args, sampler, batch_size=None):

    num_examples = int((len(linecache.getlines(args.features_file)) - 1))
    dataset = PregeneratedDataset(args.features_file, args.cache_file_dir, args.max_seq_length,
                                  num_examples, reduce_memory=True)

    dataloader = DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)

    return num_examples, dataloader
