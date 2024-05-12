import unittest

import jax
import jax.numpy as jnp

from nanodl import *


class TestDataset(unittest.TestCase):
    def test_dataset_length(self):
        class DummyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                return self.data[index]

        dataset = DummyDataset(jnp.arange(10))
        self.assertEqual(len(dataset), 10)

    def test_dataset_getitem(self):
        class DummyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                return self.data[index]

        dataset = DummyDataset(jnp.arange(10))
        item = dataset[5]
        self.assertEqual(item, 5)


class TestArrayDataset(unittest.TestCase):
    def test_array_dataset_length(self):
        dataset = ArrayDataset(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
        self.assertEqual(len(dataset), 3)

    def test_array_dataset_getitem(self):
        dataset = ArrayDataset(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
        item = dataset[1]
        self.assertEqual(item, (2, 5))


class TestDataLoader(unittest.TestCase):
    def test_data_loader_length(self):
        dataset = ArrayDataset(jnp.ones((1001, 256, 256)), jnp.ones((1001, 256, 256)))
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=False)
        self.assertEqual(len(dataloader), 101)

    def test_data_loader_iteration(self):
        dataset = ArrayDataset(jnp.ones((1001, 256, 256)), jnp.ones((1001, 256, 256)))
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)
        for a, b in dataloader:
            self.assertEqual(a.shape, (10, 256, 256))
            self.assertEqual(b.shape, (10, 256, 256))


class TestMLFunctions(unittest.TestCase):
    def test_batch_cosine_similarities(self):
        source = jnp.array([1, 0, 0])
        candidates = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        similarities = batch_cosine_similarities(source, candidates)
        expected_results = jnp.array([1.0, 0.0, 0.0])
        self.assertTrue(jnp.allclose(similarities, expected_results))

    def test_batch_pearsonr(self):
        x = jnp.array([[1, 2, 3], [4, 5, 6]])
        y = jnp.array([[6, 5, 4], [2, 6, 8]])
        correlations = batch_pearsonr(x, y)
        expected_results = jnp.array([-1.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(correlations, expected_results))

    def test_classification_scores(self):
        labels = jnp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        preds = jnp.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 0])
        scores = classification_scores(labels, preds)
        expected_results = jnp.array([0.8, 0.8, 0.8, 0.8000001])
        self.assertTrue(jnp.allclose(scores, expected_results))

    def test_mean_reciprocal_rank(self):
        predictions = jnp.array(
            [
                [0, 1, 2],  # "correct" prediction at index 0
                [1, 0, 2],  # "correct" prediction at index 1
                [2, 1, 0],  # "correct" prediction at index 2
            ]
        )
        mrr_score = mean_reciprocal_rank(predictions)
        self.assertAlmostEqual(mrr_score, 0.61111116)

    def test_jaccard(self):
        sequence1 = [1, 2, 3]
        sequence2 = [2, 3, 4]
        similarity = jaccard(sequence1, sequence2)
        self.assertAlmostEqual(similarity, 0.5)

    def test_hamming(self):
        sequence1 = jnp.array([1, 2, 3, 4])
        sequence2 = jnp.array([1, 2, 4, 4])
        similarity = hamming(sequence1, sequence2)
        self.assertEqual(similarity, 3)

    def test_zero_pad_sequences(self):
        arr = jnp.array([[1, 2, 3], [4, 5, 6]])
        max_length = 5
        padded_arr = zero_pad_sequences(arr, max_length)
        expected_padded_arr = jnp.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]])
        self.assertTrue(jnp.array_equal(padded_arr, expected_padded_arr))

    def test_entropy(self):
        probabilities = jnp.array([0.25, 0.75])
        entropy_value = entropy(probabilities)
        self.assertAlmostEqual(entropy_value, 0.8112781)

    def test_gini_impurity(self):
        probabilities = jnp.array([0.25, 0.75])
        gini_value = gini_impurity(probabilities)
        self.assertAlmostEqual(gini_value, 0.375)

    def test_kl_divergence(self):
        p = jnp.array([0.25, 0.75])
        q = jnp.array([0.5, 0.5])
        kl_value = kl_divergence(p, q)
        self.assertAlmostEqual(kl_value, 0.18872187)

    def test_count_parameters(self):
        class MyModel:
            def __init__(self):
                self.layer1 = jnp.ones((10, 20))
                self.layer2 = jnp.ones((5, 5))

        model = MyModel()
        params = model.__dict__
        total_params = count_parameters(params)
        self.assertEqual(total_params, 225)


class TestNLPFunctions(unittest.TestCase):
    def setUp(self):
        self.hypotheses = [
            "the cat is on the mat",
            "there is a cat on the mat",
        ]
        self.references = [
            "the cat is on the mat",
            "the cat sits on the mat",
        ]

    def test_rouge(self):
        rouge_scores = rouge(self.hypotheses, self.references, [1, 2])
        expected_scores = {
            "ROUGE-1": {
                "precision": 0.7857142857142857,
                "recall": 0.9,
                "f1": 0.8333333333328402,
            },
            "ROUGE-2": {
                "precision": 0.6666666666666666,
                "recall": 0.7,
                "f1": 0.6818181818176838,
            },
        }

        def assert_nested_dicts_equal(dict1, dict2):
            for key in dict1.keys():
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    assert_nested_dicts_equal(dict1[key], dict2[key])
                elif dict1[key] != dict2[key]:
                    raise AssertionError(
                        f"Values for key '{key}' are not equal: {dict1[key]} != {dict2[key]}"
                    )

        assert_nested_dicts_equal(rouge_scores, expected_scores)

    def test_bleu(self):
        bleu_score = bleu(self.hypotheses, self.references)
        self.assertAlmostEqual(bleu_score, 0.03737737833658239, places=2)

    def test_meteor(self):
        meteor_score = meteor(self.hypotheses[0], self.references[0])
        self.assertAlmostEqual(meteor_score, 1.0, places=3)  # Perfect match
        meteor_score = meteor(self.hypotheses[1], self.references[1])
        self.assertAlmostEqual(meteor_score, 0.4981684981684981, places=3)

    def test_cider_score(self):
        score = cider_score(self.hypotheses[0], self.references[0])
        self.assertAlmostEqual(score, 1.0, places=2)  # Perfect match
        score = cider_score(self.hypotheses[1], self.references[1])
        self.assertAlmostEqual(score, 0.31595617188837527, places=2)

    def test_perplexity(self):
        log_probs = [-2.3, -1.7, -0.4]
        perplexity_score = perplexity(log_probs)
        self.assertAlmostEqual(perplexity_score, 4.0, places=3)

    def test_word_error_rate(self):
        wer_score = word_error_rate(self.hypotheses, self.references)
        self.assertAlmostEqual(wer_score, 0.3333333333333333, places=2)


class TestVisionFunctions(unittest.TestCase):
    def test_normalize_images(self):
        images = jnp.array([[[[0.0, 0.5], [1.0, 0.25]]]])
        normalized_images = normalize_images(images)
        self.assertAlmostEqual(normalized_images.mean(), 0.0, places=3)
        self.assertAlmostEqual(normalized_images.std(), 1.0, places=3)

    def test_random_crop(self):
        images = jnp.ones((10, 100, 100, 3))
        crop_size = 64
        cropped_images = random_crop(images, crop_size)
        self.assertEqual(cropped_images.shape, (10, crop_size, crop_size, 3))

    def test_gaussian_blur(self):
        image = jnp.ones((5, 5, 3))
        blurred_image = gaussian_blur(image, kernel_size=3, sigma=1.0)
        self.assertEqual(blurred_image.shape, (5, 5, 3))

    def test_sobel_edge_detection(self):
        image = jnp.ones((5, 5, 3))
        edges = sobel_edge_detection(image)
        self.assertEqual(edges.shape, (5, 5))

    def test_adjust_brightness(self):
        image = jnp.ones((5, 5, 3))
        adjusted_image = adjust_brightness(image, factor=1.5)
        self.assertEqual(adjusted_image.shape, (5, 5, 3))

    def test_adjust_contrast(self):
        image = jnp.ones((5, 5, 3))
        adjusted_image = adjust_contrast(image, factor=1.5)
        self.assertEqual(adjusted_image.shape, (5, 5, 3))

    def test_flip_image(self):
        image = jnp.ones((5, 5, 3))
        flipped_image_horizontally = flip_image(image, jnp.array([True]))
        flipped_image_vertically = flip_image(image, jnp.array([False]))
        self.assertEqual(flipped_image_horizontally.shape, (5, 5, 3))
        self.assertEqual(flipped_image_vertically.shape, (5, 5, 3))

    def test_random_flip_image(self):
        key = jax.random.PRNGKey(0)
        image = jnp.ones((5, 5, 3))
        flipped_image = random_flip_image(image, key, jnp.array([True]))
        self.assertEqual(flipped_image.shape, (5, 5, 3))


if __name__ == "__main__":
    unittest.main()
