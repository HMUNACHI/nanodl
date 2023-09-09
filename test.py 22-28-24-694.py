"""
# attention test
from jax import random
key = random.PRNGKey(0)
main_rng, x_rng = random.split(key)
x = random.normal(x_rng, (3, 16, 128))
mh_attn = CrossMultiHeadAttention(hidden_dim=128, num_heads=4)
main_rng, init_rng = random.split(main_rng)
params = mh_attn.init(init_rng, x, x)['params']
# Apply attention with parameters on the inputs
out, attn = mh_attn.apply({'params': params}, x, x)
print('Out', out.shape, 'Attention', attn.shape)


# Transformer encoder test
from jax import random
key = random.PRNGKey(0)
main_rng, x_rng = random.split(key)
x = random.normal(x_rng, (3, 16, 128))
encblock = TransformerEncoder(input_dim=128, num_heads=4, num_layers=2, feedforward_dim=256, dropout=0.2)
# Initialize parameters of encoder block with random key and inputs
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = encblock.init({'params': init_rng, 'dropout': dropout_init_rng}, x, training=True)['params']
# Apply encoder block with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
main_rng, dropout_apply_rng = random.split(main_rng)
out, att1 = encblock.apply({'params': params}, x, training=True, rngs={'dropout': dropout_apply_rng})
print('Out', out.shape, att1.shape)


# Transformer decoder test
from jax import random
key = random.PRNGKey(0)
main_rng, x_rng = random.split(key)
x = random.normal(x_rng, (3, 16, 128))
encblock = TransformerDecoder(input_dim=128, num_heads=4, num_layers=2, feedforward_dim=256, dropout=0.2)
# Initialize parameters of encoder block with random key and inputs
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = encblock.init({'params': init_rng, 'dropout': dropout_init_rng}, x, x, training=True)['params']
# Apply encoder block with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
main_rng, dropout_apply_rng = random.split(main_rng)
out, att1, att2 = encblock.apply({'params': params}, x, x, training=True, rngs={'dropout': dropout_apply_rng})
print('Out', out.shape, att1.shape, att2.shape)


# Vit test
from jax import random
key = random.PRNGKey(0)
main_rng, x_rng = random.split(key)
x = random.normal(x_rng, (5, 256, 256, 3))
patch_size = (16, 16)
input_dim = int((x.shape[1] * x.shape[2]) / (patch_size[0] * patch_size[1]))

model = ViT(patch_size=patch_size,
            num_layers=2,
            input_dim=input_dim, 
            num_heads=4, 
            feedforward_dim=256, 
            dropout=0.2,
            n_outputs=100)

main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = model.init({'params': init_rng, 'dropout': dropout_init_rng}, x, training=True)['params']
main_rng, dropout_apply_rng = random.split(main_rng)
out, att1 = model.apply({'params': params}, x, training=True, rngs={'dropout': dropout_apply_rng})
print('Out', out.shape, att1.shape)

"""