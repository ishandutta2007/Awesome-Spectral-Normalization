# Singular Value Bounding (SVB) / Relaxed SN

Singular Value Bounding (SVB) or Relaxed Spectral Normalization is a regularization technique that restricts the singular values of the weight matrices to a specific bounding range, rather than forcing the largest singular value to be exactly 1.0.

## Mechanism
Instead of projecting the spectral norm directly to 1, SVB constrains the singular values $\sigma(W)$ to lie within a relaxed interval $[c_{\text{min}}, c_{\text{max}}]$:
$$\sigma_i(W) \leftarrow \max(c_{\text{min}}, \min(c_{\text{max}}, \sigma_i(W)))$$
This enforces near-orthogonality and keeps the Lipschitz constant bounded while retaining parameter flexibility.

```mermaid
flowchart TD
    A[Compute Singular Values of W] --> B{Are singular values in range [c_min, c_max]?}
    B -- Yes --> C[Keep weights unchanged]
    B -- No --> D[Clip singular values to [c_min, c_max]]
    D --> E[Reconstruct weight matrix W via SVD]
```

## Advantages
- **Representational Capacity:** Allows the model to learn functions with Lipschitz constants greater than 1 if needed, preventing over-smoothing.
- **Orthogonality:** Encourages weights to be close to orthogonal matrices, stabilizing training in very deep networks.

## Limitations
- **Optimization Cost:** Often requires performing SVD or partial SVD frequently, which can be computationally heavy during training.

## References
- Jia, K., Tao, D., Gao, S., & Xu, X. (2017). [Improving Training of Deep Neural Networks via Singular Value Bounding](https://arxiv.org/abs/1611.06013).
