from functools import reduce


class Grid:
    def __init__(self, grid: dict):
        self._keys = list(grid.keys())
        self._values = list(grid.values())
        self._combinations = []

    def __iter__(self):
        self._combinations = [[value] for value in self._values[0]]
        if len(self._values) > 1:
            for val_list in self._values[1:]:
                self._combinations = [comb + [value] for value in val_list for comb in self._combinations]

        return self

    def __next__(self):
        if len(self._combinations) > 0:
            return dict(zip(self._keys, self._combinations.pop(0)))
        else:
            raise StopIteration()

    def __len__(self):
        return reduce(lambda acc, item: acc * len(item), self._values, 1)
