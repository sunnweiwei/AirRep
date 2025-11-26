"""Data sampling utilities for creating subset-dev pairs."""

import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm


class SubsetDevSampler:
    """Create cross-validation splits and sample subset-dev pairs."""

    def __init__(
        self,
        dev_size: int = 10000,
        train_size: int = 100000,
        n_splits: int = 100,
        subset_size: int = 1000,
        n_subsets_per_split: int = 100,
        seed: int = 42
    ):
        """
        Initialize sampler.

        Args:
            dev_size: Size of development set
            train_size: Size of training set (after dev split)
            n_splits: Number of cross-validation splits
            subset_size: Size of each training subset
            n_subsets_per_split: Number of subsets to sample per split
            seed: Random seed
        """
        self.dev_size = dev_size
        self.train_size = train_size
        self.n_splits = n_splits
        self.subset_size = subset_size
        self.n_subsets_per_split = n_subsets_per_split
        self.seed = seed
        np.random.seed(seed)

    def sample(self, data: List[Any]) -> List[Dict]:
        """
        Sample subset-dev pairs from data.

        Args:
            data: List of training examples

        Returns:
            List of dicts with keys: {id, select, dev, dev_id}
        """
        if len(data) < self.dev_size + self.train_size:
            raise ValueError(
                f"Data size {len(data)} too small for dev_size={self.dev_size} "
                f"+ train_size={self.train_size}"
            )

        index = list(range(len(data)))
        index_data = []
        dev_seen = set()
        set_seen = set()

        print(f"Sampling {self.n_splits} splits with {self.n_subsets_per_split} subsets each...")

        for split_id in tqdm(range(self.n_splits)):
            # Shuffle and create dev-train split
            np.random.shuffle(index)
            dev_set = sorted(index[:self.dev_size])

            # Skip if dev set already seen
            dev_key = str(dev_set)
            if dev_key in dev_seen:
                continue
            dev_seen.add(dev_key)

            train_set = index[self.dev_size:self.dev_size + self.train_size]

            # Sample subsets from train set
            subset_count = 0
            while subset_count < self.n_subsets_per_split:
                np.random.shuffle(train_set)
                select_set = sorted(train_set[:self.subset_size])

                # Skip if subset already seen
                select_key = str(select_set)
                if select_key in set_seen:
                    continue
                set_seen.add(select_key)

                # Create subset-dev pair
                index_data.append({
                    'id': f"{split_id}-{subset_count}",
                    'select': select_set,
                    'dev': dev_set,
                    'dev_id': split_id
                })

                subset_count += 1

        print(f"Generated {len(index_data)} subset-dev pairs")
        return index_data