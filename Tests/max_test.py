from prepareDataset import buildDataframe
import pandas as pd
import unittest
from subGenreModel import SubGenreClassifier
from genreModel import topGenreClassifier
from featureModel import audioFeatureModel
import torch

class TestSubGenreClassifierOutputShape(unittest.TestCase):
    def test_sub_genre_output_shape(self):
        num_classes = 16
        model = SubGenreClassifier(num_classes = num_classes)
        input_tensor = torch.randn(1, 2, 142, 2580)
        output = model(input_tensor)
        expected_shape = torch.Size([1, num_classes])
        self.assertEqual(output.shape, expected_shape)

    def test_top_genre_output_shape(self):
        model = topGenreClassifier()
        input_tensor = torch.randn(10, 2, 142, 2580)
        output = model(input_tensor)
        expected_shape = torch.Size([10, 16])
        self.assertEqual(output.shape, expected_shape)

    def test_feature_output_shape(self):
        model = audioFeatureModel()
        input_tensor = torch.randn(10, 2, 142, 2580)
        output = model(input_tensor)
        expected_shape = torch.Size([10, 8])
        self.assertEqual(output.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
