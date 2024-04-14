import unittest
from unittest.mock import MagicMock
import torch
from fmaDataset import fmaDataset

class TestFMADataset(unittest.TestCase):
    def setUp(self):
        self.dataframe = MagicMock()
        self.beats = [(torch.tensor(95.7031, dtype=torch.float64), torch.tensor([3, 43, 84, 137], dtype=torch.int32)) for _ in range(5)]
        self.spectrogram = [torch.rand(2, 10, 2580) for _ in range(5)]
        self.mfcc = [torch.rand(2, 10, 2580) for _ in range(5)]
        self.labels = torch.randint(0, 2, (5,))
        self.ids = list(range(5))
        self.dataset = fmaDataset(self.dataframe, self.beats, self.spectrogram,
                                  self.mfcc, self.labels, self.ids)

    def test_initialization(self):
        self.assertEqual(self.dataset.dataframe, self.dataframe)
        self.assertEqual(self.dataset.beats, self.beats)
        self.assertEqual(self.dataset.spectrogram, self.spectrogram)
        self.assertEqual(self.dataset.mfcc, self.mfcc)
        self.assertTrue(torch.equal(self.dataset.labels, self.labels))
        self.assertEqual(self.dataset.id, self.ids)

    def test_getitem(self):
        combined_features, label = self.dataset[0]
        self.assertEqual(combined_features.shape, (2, 21, 2580))
        self.assertEqual(label, self.labels[0])

if __name__ == '__main__':
    unittest.main()