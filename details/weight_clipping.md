# Weight Clipping Precursor

Weight clipping was the original method proposed in the Wasserstein GAN (WGAN) paper to enforce the **1-Lipschitz continuity** constraint on the critic (discriminator).

## Mathematical Formulation
In a WGAN, the critic $f_w$ is parameterized by weights $w$. To satisfy the Lipschitz condition, the weights are restricted to a compact space:
$$w \in [-c, c]^d$$
where $c$ is a small hyperparameter (e.g., $c = 0.01$). After every gradient update, the weights are clipped:
$$w \leftarrow \text{clip}(w, -c, c)$$

```mermaid
flowchart TD
    A[Gradient Update] --> B[Compute w_new = w - eta * grad]
    B --> C{w_new in [-c, c]?}
    C -- No --> D[Clip w_new to range [-c, c]]
    C -- Yes --> E[Keep w_new]
    D --> F[Final Weights]
    E --> F
```

## Advantages
- Extremely simple to implement.
- Does not require computing second-order derivatives or complex matrix operations.
- Enabled the first stable training of Wasserstein GANs without severe mode collapse.

## Limitations
- **Capacity Collapse:** Restricting weights to a small hypercube forces the network to learn overly simplified functions.
- **Gradient Exploding/Vanishing:** If $c$ is too large, gradients can explode; if $c$ is too small, gradients vanish as they propagate through deep layers.
- **Capacity Redundancy:** The network tends to push weights to the extremes ($-c$ or $c$), meaning only a fraction of the capacity is effectively used.

## References
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). [Wasserstein GAN](https://arxiv.org/abs/1701.07875).
