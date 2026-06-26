# The Power Iteration Method (Online Approximation)

The Power Iteration Method is the numerical algorithm used in Spectral Normalization to efficiently approximate the largest singular value (spectral norm) of a matrix without running a full Singular Value Decomposition (SVD).

## Mathematical Formulation
For a weight matrix $W$, the power iteration method maintains two vectors, $u$ (left eigenvector approximation) and $v$ (right eigenvector approximation).
In every forward pass step, the vectors are updated iteratively:
1. $v \leftarrow \frac{W^T u}{\|W^T u\|_2}$
2. $u \leftarrow \frac{W v}{\|W v\|_2}$

The spectral norm $\sigma(W)$ is then approximated as:
$$\sigma(W) \approx u^T W v$$

Usually, running just **one iteration** per training step is sufficient, as the weights change slowly during optimization.

```mermaid
flowchart TD
    A[Current Vectors u, v] --> B[Calculate v = W^T u / ||W^T u||]
    B --> C[Calculate u = W v / ||W v||]
    C --> D[Estimate sigma = u^T W v]
    D --> E[Scale Weights: W_SN = W / sigma]
    E --> F[Update u, v for the next step]
```

## Advantages
- **Extremely Fast:** Computational complexity is $O(N^2)$ (matrix-vector multiplication) compared to $O(N^3)$ for full SVD.
- **Differentiable:** The normalization step is easily integrated into standard automatic differentiation graphs.

## References
- Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957).
