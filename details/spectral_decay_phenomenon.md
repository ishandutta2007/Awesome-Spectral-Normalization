# The Spectral Decay Phenomenon

The Spectral Decay Phenomenon refers to the undesirable behavior where the non-dominant singular values of a weight matrix decay toward zero over long training runs under spectral normalization.

## The Problem
While spectral normalization constrains the largest singular value ($\sigma_1 = 1$), it does not prevent the remaining singular values ($\sigma_2, \sigma_3, \dots$) from decaying. As a result:
$$\text{Rank}(W) \rightarrow 1$$
This rank collapse makes the weight matrix redundant, reducing the effective parameters of the layer and causing representation bottlenecks.

```mermaid
flowchart TD
    A[Normal Weight Matrix Rank] --> B[Spectral Normalization applied]
    B --> C[Long Training Runs]
    C --> D[Non-dominant Singular Values decay to 0]
    D --> E[Rank Collapse & Parameter Redundancy]
```

## Mitigation: Orthogonal Regularization
To prevent this, SN is combined with Orthogonal Regularization, which encourages the weight matrix to remain orthogonal (preserving full rank):
$$L_{\text{orth}} = \lambda \|W^T W - I\|_F^2$$
This ensures that all singular values remain close to 1.0, preserving layer capacity.

```mermaid
flowchart TD
    A[Add Orthogonal Regularization to Loss] --> B[Encourage W^T W approx I]
    B --> C[All singular values remain close to 1]
    C --> D[Prevents Rank Collapse]
```

## References
- Brock, A., Donahue, J., & Simonyan, K. (2018). [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096).
