import jax
import jax.numpy as jnp
from flax import linen as nn


class GraphAttentionLayer(nn.Module):
    """
    A single graph attention layer as part of a Graph Attention Network (GAT).

    This layer applies a self-attention mechanism on the nodes of a graph. Each node's features are transformed through a learned linear transformation, and attention coefficients are computed to determine the importance of every other node's features. This allows the model to dynamically adjust which nodes contribute most to the next layer's input for each node.

    Attributes:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per node.
        dropout_rate (float): Dropout rate applied to features and attention coefficients for regularization.
        alpha (float): Negative slope coefficient for the LeakyReLU activation function used in computing attention scores.
        concat (bool, optional): Whether to concatenate the output of attention heads in a multi-head attention mechanism. Default is True.

    Methods:
        __call__(self, x: jnp.ndarray, adj: jnp.ndarray, training: bool) -> jnp.ndarray:
            Forward pass of the graph attention layer.

            Args:
                x (jnp.ndarray): The input node features, shape (N, in_features), where N is the number of nodes.
                adj (jnp.ndarray): The adjacency matrix of the graph, shape (N, N), indicating node connections.
                training (bool): Whether the layer is being used in training mode. Affects dropout behavior.

            Returns:
                jnp.ndarray: The output node features after the attention mechanism. If `concat` is True, applies a non-linearity (LeakyReLU); otherwise, returns the linear combination of features directly. Shape is (N, out_features).
    """

    in_features: int
    out_features: int
    dropout_rate: float
    alpha: float
    concat: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray, training: bool) -> jnp.ndarray:

        W = self.param(
            "W",
            jax.nn.initializers.glorot_uniform(),
            (self.in_features, self.out_features),
        )

        a = self.param(
            "a", jax.nn.initializers.glorot_uniform(), (2 * self.out_features, 1)
        )

        h = jnp.dot(x, W)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)

        N = h.shape[0]
        a_input = jnp.concatenate(
            [h[:, None, :].repeat(N, axis=1), h[None, :, :].repeat(N, axis=0)], axis=2
        )

        e = nn.leaky_relu(jnp.dot(a_input, a).squeeze(-1), negative_slope=self.alpha)

        zero_vec = -9e15 * jnp.ones_like(e)
        attention = jnp.where(adj > 0, e, zero_vec)
        attention = nn.softmax(attention, axis=1)

        attention = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(
            attention
        )

        h_prime = jnp.matmul(attention, h)

        if self.concat:
            return nn.leaky_relu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    """
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

    Attributes:
        nfeat (int): Number of features for each node in the input graph.
        nhid (int): Number of hidden units in each graph attention layer.
        nclass (int): Number of classes for the node classification task.
        dropout_rate (float): Dropout rate for regularization. Applied to the inputs of each graph attention layer and the final output.
        alpha (float): LeakyReLU angle of negative slope used in the attention mechanism.
        nheads (int): Number of attention heads. Each head computes a separate attention mechanism over the input, and their results are concatenated.

    Methods:
        __call__(self, x: jnp.ndarray, adj: jnp.ndarray, training: bool = False) -> jnp.ndarray:
            Forward pass of the GAT model.

            Args:
                x (jnp.ndarray): Node features matrix with shape (N, nfeat), where N is the number of nodes in the graph.
                adj (jnp.ndarray): Adjacency matrix of the graph with shape (N, N). It should represent the graph structure.
                training (bool, optional): Flag to indicate whether the model is being used for training. Affects dropout behavior. Defaults to False.

            Returns:
                jnp.ndarray: The output node features after passing through the GAT model. Shape is (N, nclass), representing the class scores for each node.
    """

    nfeat: int
    nhid: int
    nclass: int
    dropout_rate: float
    alpha: float
    nheads: int

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, adj: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:

        heads = [
            GraphAttentionLayer(
                self.nfeat,
                self.nhid,
                dropout_rate=self.dropout_rate,
                alpha=self.alpha,
                concat=True,
            )
            for _ in range(self.nheads)
        ]

        x = jnp.concatenate([head(x, adj, training) for head in heads], axis=1)

        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        out_att = GraphAttentionLayer(
            self.nhid * self.nheads,
            self.nclass,
            dropout_rate=self.dropout_rate,
            alpha=self.alpha,
            concat=False,
        )

        return out_att(x, adj, training)
