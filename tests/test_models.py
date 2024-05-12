import unittest

import jax
import jax.numpy as jnp

from nanodl import *


class TestTextBasedModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.max_length = 51
        self.vocab_size = 1000
        self.embed_dim = 256

        self.data = jnp.arange(
            self.batch_size * self.max_length, dtype=jnp.int32
        ).reshape((self.batch_size, self.max_length))

        self.dummy_inputs = self.data[:, :-1]
        self.dummy_targets = self.data[:, 1:]

        self.hyperparams = {
            "num_layers": 1,
            "hidden_dim": self.embed_dim,
            "num_heads": 2,
            "feedforward_dim": self.embed_dim,
            "dropout": 0.1,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "max_length": self.max_length,
            "start_token": 0,
            "end_token": 50,
        }

    def test_t5_model(self):
        model = T5(**self.hyperparams)
        self._test_encoder_decoder_model(model)

    def test_transformer_model(self):
        model = Transformer(**self.hyperparams)
        self._test_encoder_decoder_model(model)

    def test_lamda_model(self):
        model = LaMDA(**self.hyperparams)
        self._test_decoder_only_model(model)

    def test_gpt3_model(self):
        model = GPT4(**self.hyperparams)
        self._test_decoder_only_model(model)

    def test_gpt3_model(self):
        model = GPT4(**self.hyperparams)
        self._test_decoder_only_model(model)

    def test_mistral_model(self):
        model = Mistral(**self.hyperparams, num_groups=2, window_size=5, shift_size=2)
        self._test_decoder_only_model(model)

    def test_mixtral_model(self):
        model = Mixtral(**self.hyperparams, num_groups=2, window_size=5, shift_size=2)
        self._test_decoder_only_model(model)

    def test_llama_model(self):
        model = Llama3(**self.hyperparams, num_groups=2)
        self._test_decoder_only_model(model)

    def test_gemma_model(self):
        model = Gemma(**self.hyperparams, num_groups=2)
        self._test_decoder_only_model(model)

    def _test_encoder_decoder_model(self, model):
        rngs = {"params": jax.random.key(0), "dropout": jax.random.key(1)}
        params = model.init(rngs, self.dummy_inputs, self.dummy_targets)["params"]

        outputs = model.apply(
            {"params": params}, self.dummy_inputs, self.dummy_targets, rngs=rngs
        )

        self.assertEqual(
            outputs.shape, (self.batch_size, self.max_length - 1, self.vocab_size)
        )

    def _test_decoder_only_model(self, model):
        rngs = {"params": jax.random.key(0), "dropout": jax.random.key(1)}
        params = model.init(rngs, self.dummy_inputs)["params"]

        outputs = model.apply({"params": params}, self.dummy_inputs, rngs=rngs)

        self.assertEqual(
            outputs.shape, (self.batch_size, self.max_length - 1, self.vocab_size)
        )

    def test_reward_model(self):
        model = RewardModel(
            Mixtral(**self.hyperparams, num_groups=2, window_size=5, shift_size=2),
            dim=self.hyperparams["hidden_dim"],
            dropout=0.1,
        )
        rngs = jax.random.PRNGKey(0)
        rngs, dropout_rng = jax.random.split(rngs)
        params = model.init(
            {"params": rngs, "dropout": dropout_rng}, self.dummy_inputs
        )["params"]
        rewards = model.apply(
            {"params": params}, self.dummy_inputs, rngs={"dropout": dropout_rng}
        )
        assert rewards.shape == (self.batch_size,)


class TestVisionBasedModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.n_outputs = 5
        self.embed_dim = 256
        self.patch_size = (16, 16)
        self.dummy_inputs = jnp.ones((self.batch_size, 224, 224, 3))
        key = jax.random.PRNGKey(10)

        self.dummy_labels = jax.random.randint(
            key, shape=(self.batch_size,), minval=0, maxval=self.n_outputs - 1
        )

        self.hyperparams = {
            "dropout": 0.1,
            "num_heads": 2,
            "feedforward_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "hidden_dim": self.embed_dim,
            "num_layers": 4,
            "n_outputs": self.n_outputs,
        }

    def test_vit_model(self):
        model = ViT(**self.hyperparams)
        self._test_model(model)

    def test_mixer_model(self):
        model = Mixer(**self.hyperparams)
        self._test_model(model)

    def _test_model(self, model):
        rngs = {"params": jax.random.key(0), "dropout": jax.random.key(1)}

        params = model.init(rngs, self.dummy_inputs)["params"]

        outputs = model.apply({"params": params}, self.dummy_inputs, rngs=rngs)[0]

        self.assertEqual(outputs.shape, (self.batch_size, self.n_outputs))


class TestCLIPModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.max_length = 50
        self.vocab_size = 1000
        self.embed_dim = 256
        self.dummy_texts = jnp.ones((self.batch_size, self.max_length), dtype=jnp.int32)
        self.dummy_images = jnp.ones((self.batch_size, 224, 224, 3))

        self.clip_params = {
            "dropout": 0.1,
            "num_heads": 8,
            "feedforward_dim": self.embed_dim,
            "num_layers_text": 4,
            "hidden_dim_text": self.embed_dim,
            "image_patch_size": (16, 16),
            "hidden_dim_image": self.embed_dim,
            "num_layers_images": 4,
            "max_len": self.max_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        }

        self.model = CLIP(**self.clip_params)

    def test_clip_model_initialization_and_processing(self):
        rng = jax.random.PRNGKey(0)
        params = self.model.init(rng, self.dummy_texts, self.dummy_images)["params"]

        loss = self.model.apply({"params": params}, self.dummy_texts, self.dummy_images)

        self.assertIsNotNone(loss)


class TestWhisperModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.max_length = 50
        self.embed_dim = 256
        self.vocab_size = 1000

        self.dummy_targets = jnp.arange(
            self.batch_size * self.max_length, dtype=jnp.int32
        ).reshape((self.batch_size, self.max_length))

        self.dummy_inputs = jnp.ones((self.batch_size, self.max_length, self.embed_dim))

        self.hyperparams = {
            "num_layers": 1,
            "hidden_dim": self.embed_dim,
            "num_heads": 2,
            "feedforward_dim": self.embed_dim,
            "dropout": 0.1,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "max_length": self.max_length,
            "start_token": 0,
            "end_token": 50,
        }

        self.model = Whisper(**self.hyperparams)

    def test_whisper_model_initialization_and_processing(self):
        rngs = {"params": jax.random.key(0), "dropout": jax.random.key(1)}

        params = self.model.init(rngs, self.dummy_inputs, self.dummy_targets)["params"]

        outputs = self.model.apply(
            {"params": params}, self.dummy_inputs, self.dummy_targets, rngs=rngs
        )

        self.assertEqual(
            outputs.shape, (self.batch_size, self.max_length, self.vocab_size)
        )


class TestDiffusionModel(unittest.TestCase):
    def setUp(self):
        self.image_size = 32
        self.widths = [32, 64, 128]
        self.block_depth = 2
        self.input_shape = (3, self.image_size, self.image_size, 3)
        self.images = jax.random.normal(jax.random.PRNGKey(0), self.input_shape)

        self.model = DiffusionModel(self.image_size, self.widths, self.block_depth)

    def test_diffusion_model_initialization_and_processing(self):
        params = self.model.init(jax.random.PRNGKey(0), self.images)
        pred_noises, pred_images = self.model.apply(params, self.images)
        self.assertEqual(pred_noises.shape, self.input_shape)
        self.assertEqual(pred_images.shape, self.input_shape)


class TestGATModel(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 10
        self.num_features = 5
        self.nclass = 3

        self.x = jax.random.normal(
            jax.random.PRNGKey(0), (self.num_nodes, self.num_features)
        )

        self.adj = jax.random.bernoulli(
            jax.random.PRNGKey(0), 0.3, (self.num_nodes, self.num_nodes)
        )

        self.model = GAT(
            nfeat=self.num_features,
            nhid=8,
            nclass=self.nclass,
            dropout_rate=0.5,
            alpha=0.2,
            nheads=3,
        )

    def test_gat_model_initialization_and_processing(self):
        params = self.model.init(jax.random.key(0), self.x, self.adj, training=False)

        output = self.model.apply(params, self.x, self.adj, training=False)

        self.assertEqual(output.shape, (self.num_nodes, self.nclass))


class TestIJEPAModel(unittest.TestCase):
    def setUp(self):
        self.image_size = 128
        self.num_channels = 3
        self.patch_size = 16
        self.embed_dim = 32
        self.predictor_bottleneck = 16
        self.num_heads = 4
        self.predictor_num_heads = 4
        self.num_layers = 2
        self.predictor_num_layers = 1
        self.dropout_p = 0
        self.num_patches = (self.image_size**2) / (self.patch_size**2)

        self.x = jax.random.normal(
            jax.random.PRNGKey(0),
            (1, self.image_size, self.image_size, self.num_channels),
        )

        self.model = IJEPA(
            image_size=self.image_size,
            num_channels=self.num_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            predictor_bottleneck=self.predictor_bottleneck,
            num_heads=self.num_heads,
            predictor_num_heads=self.predictor_num_heads,
            num_layers=self.num_layers,
            predictor_num_layers=self.predictor_num_layers,
            dropout_p=self.dropout_p,
        )

        self.data_sampler = IJEPADataSampler(
            image_size=self.image_size, M=4, patch_size=self.patch_size
        )

    def test_ijepa_data_sampling(self):
        context_mask, target_mask = self.data_sampler()
        self.assertEqual(context_mask.shape, (4, self.num_patches))
        self.assertEqual(target_mask.shape, (4, self.num_patches))

    def test_ijepa_model_initialization_and_processing(self):
        context_mask, target_mask = self.data_sampler()

        params = self.model.init(
            jax.random.key(0),
            self.x,
            context_mask[jnp.newaxis],
            target_mask[jnp.newaxis],
            training=False,
        )

        outputs, _ = self.model.apply(
            params,
            self.x,
            context_mask[jnp.newaxis],
            target_mask[jnp.newaxis],
            training=False,
        )

        self.assertEqual(len(outputs), 4)
        self.assertEqual(outputs[0][0].shape, (1, self.num_patches, self.embed_dim))
        self.assertEqual(outputs[0][0].shape, outputs[0][1].shape)


if __name__ == "__main__":
    unittest.main()
