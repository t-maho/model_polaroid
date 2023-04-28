import numpy as np
from torch.utils.data import Dataset


class GridDataset(Dataset):
    def __init__(
        self, 
        origin, 
        direction1, 
        direction2,
        n_steps=50,
        max_step=1):

        self.origin = origin
        self.direction1 = direction1
        self.direction2 = direction2

        self.set_grid(n_steps, max_step)
        self.step_size = abs(self.grid[1] - self.grid[0])

    def __len__(self):
        return len(self.grid) ** 2

    def set_grid(self, n_steps, max_step):
        self.grid = np.linspace(-max_step, max_step, n_steps)

    def __getitem__(self, idx):
        j = int(idx / len(self.grid))
        i = int(idx % len(self.grid))
        
        # print((self.direction1.direction * self.direction2.direction).sum())
        image = self.origin + self.grid[i] * self.direction1.direction + self.grid[j] * self.direction2.direction

        # print(i, j, self.grid[i], self.grid[j], image.max(), image.min())
        return image