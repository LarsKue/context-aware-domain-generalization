
from torch.utils.data import Dataset, Subset


class DynamicCachedDataset(Dataset):
    def __init__(self, dataset: Dataset, size: int = 10_000, chunk_size: int = 5_000):
        """
        @param dataset: The dataset to cache.
        @param size: The maximum number of items to cache.
        @param chunk_size: The number of items to drop from the cache when it is full. This is to avoid thrashing.
            We recommend setting this to around half the size of the cache.
        """
        if size < chunk_size:
            raise ValueError(f"Total cache size {size} must be greater than chunk size {chunk_size}")

        self.dataset = dataset
        self.size = size
        self.chunk_size = chunk_size

        self.cache = {}
        self.cache_queue = []

    def add_to_cache(self, item):
        self.cache[item] = self.dataset[item]
        self.cache_queue.append(item)

        if len(self.cache_queue) > self.size:
            # drop the first chunk_size items
            drop = self.cache_queue[:self.chunk_size]
            for item in drop:
                self.cache.pop(item)
            self.cache_queue = self.cache_queue[self.chunk_size:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if item not in self.cache:
            self.add_to_cache(item)

        return self.cache[item]

    @property
    def in_distribution(self):
        return self.dataset.in_distribution

    def domain(self, domain: int) -> Subset:
        return Subset(self, self.dataset.domain_indices(domain))


class StaticCachedDataset(Dataset):
    def __init__(self, dataset: Dataset, size: int = 10_000):
        super().__init__()
        self.dataset = dataset
        self.size = size
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if item not in self.cache:
            if len(self.cache) < self.size:
                self.cache[item] = self.dataset[item]
            else:
                return self.dataset[item]

        return self.cache[item]

    @property
    def in_distribution(self):
        return self.dataset.in_distribution

    def domain(self, domain: int) -> Subset:
        return Subset(self, self.dataset.domain_indices(domain))
