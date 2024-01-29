'''
Graph Attention Networks (GATs) are a type of neural network designed for graph-structured data. 
The key feature of GATs is the use of attention mechanisms to weigh the importance of nodes' neighbors. 
This allows GATs to focus on the most relevant parts of the graph structure when learning node representations. 
In GATs, each node aggregates information from its neighbors, but not all neighbors contribute equally. 
The attention mechanism computes weights that determine the importance of each neighbor's features to the target node. 
These weights are learned during training and are based on the features of the nodes involved.
GATs can handle graphs with varying sizes and connectivity patterns, making them suitable for a wide range of applications, 
including social network analysis, recommendation systems, and molecular structure analysis.

Example usage:
```
import jax
import jax.numpy as jnp
from nanodl import ArrayDataset, DataLoader
from nanodl import GAT

# Generate dummy data
batch_size = 8
max_length = 10
nclass = 3

# Replace with actual tokenised data
# Generate a random key for Jax
key = jax.random.PRNGKey(0)
num_nodes = 10
num_features = 5
x = jax.random.normal(key, (num_nodes, num_features))  # Features for each node
adj = jax.random.bernoulli(key, 0.3, (num_nodes, num_nodes))  # Random adjacency matrix

# Initialize the GAT model
model = GAT(nfeat=num_features, 
            nhid=8, 
            nclass=nclass, 
            dropout_rate=0.5, 
            alpha=0.2, 
            nheads=3)

# Initialize the model parameters
params = model.init(key, x, adj)
output = model.apply(params, x, adj)
print("Output shape:", output.shape)
```
'''

import jax, flax, optax, time
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from typing import Any, Tuple, Optional, Iterable

class GraphAttentionLayer(nn.Module):
    in_features: int
    out_features: int
    dropout_rate: float
    alpha: float
    concat: bool = True

    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray, 
                 adj: jnp.ndarray, 
                 training: bool) -> jnp.ndarray:
        """
        Forward pass for Graph Attention Layer.
        
        Args:
            x (jnp.ndarray): Node feature matrix (N, in_features).
            adj (jnp.ndarray): Adjacency matrix (N, N).
            training (bool): If True, the dropout is applied.

        Returns:
            jnp.ndarray: Output feature matrix (N, out_features).
        """

        # Initialize weights
        W = self.param('W', jax.nn.initializers.glorot_uniform(), 
                       (self.in_features, self.out_features))
        
        a = self.param('a', jax.nn.initializers.glorot_uniform(), 
                       (2 * self.out_features, 1))

        # Apply linear transformation
        h = jnp.dot(x, W)

        # Apply dropout if not deterministic
        h = nn.Dropout(rate=self.dropout_rate, 
                       deterministic=not training)(h)

        # Attention mechanism
        N = h.shape[0]
        a_input = jnp.concatenate([h[:, None, :].repeat(N, axis=1), 
                                   h[None, :, :].repeat(N, axis=0)], axis=2)
        
        e = nn.leaky_relu(jnp.dot(a_input, a).squeeze(-1), 
                          negative_slope=self.alpha)

        # Masked attention
        zero_vec = -9e15 * jnp.ones_like(e)
        attention = jnp.where(adj > 0, e, zero_vec)
        attention = nn.softmax(attention, axis=1)

        attention = nn.Dropout(rate=self.dropout_rate, 
                               deterministic=not training)(attention)

        # Apply attention and concatenate
        h_prime = jnp.matmul(attention, h)

        if self.concat:
            return nn.leaky_relu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    nfeat: int
    nhid: int
    nclass: int
    dropout_rate: float
    alpha: float
    nheads: int

    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray, 
                 adj: jnp.ndarray, 
                 training: bool = False) -> jnp.ndarray:
        """
        Forward pass for Graph Attention Network (GAT).
        
        Args:
            x (jnp.ndarray): Node feature matrix (N, nfeat).
            adj (jnp.ndarray): Adjacency matrix (N, N).
            Training (bool): If True, the dropout is applied.

        Returns:
            jnp.ndarray: Log softmax output for node classification (N, nclass).
        """

        # Apply multiple graph attention layers (heads)
        heads = [GraphAttentionLayer(self.nfeat, 
                                     self.nhid, 
                                     dropout_rate=self.dropout_rate, 
                                     alpha=self.alpha, concat=True) for _ in range(self.nheads)]
        
        x = jnp.concatenate([head(x, adj, training) for head in heads], axis=1)
        
        # Apply dropout if not deterministic
        x = nn.Dropout(rate=self.dropout_rate, 
                       deterministic=not training)(x)

        # Apply output graph attention layer
        out_att = GraphAttentionLayer(self.nhid * self.nheads, 
                                      self.nclass, 
                                      dropout_rate=self.dropout_rate, 
                                      alpha=self.alpha, concat=False)
        
        return out_att(x, adj, training)