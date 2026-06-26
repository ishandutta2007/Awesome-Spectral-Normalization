# Adversarial Defense Hardening & Certified Robustness

Spectral Normalization is a critical tool for creating networks with certified robustness against adversarial attacks, such as Fast Gradient Sign Method (FGSM) or Projected Gradient Descent (PGD).

## Mechanism
Adversarial attacks exploit the high sensitivity of neural networks by adding small perturbations $\delta$ to the input $x$. If the network has a large Lipschitz constant $L$, a small change in input can trigger a huge change in output:
$$\|f(x + \delta) - f(x)\| \le L \|\delta\|$$
By applying Spectral Normalization, the Lipschitz constant $L$ is bounded (e.g., $L \le 1$). This guarantees that the network's output remains stable for any input perturbation within a certain radius, providing certified robustness.

```mermaid
flowchart LR
    A[Input x + delta] --> B[1-Lipschitz Network (SN)]
    B --> C[Stable Output Change <= ||delta||]
    C --> D[Resilience to Adversarial Attacks]
```

## References
- Cisse, M., Bojanowski, P., Grave, E., Dauphin, Y., & Usunier, N. (2017). [Parseval Networks: Improving Robustness to Adversarial Examples](https://arxiv.org/abs/1704.08847).
- Farnia, F., Zhang, J., & Tse, D. (2019). [Generalizable Adversarial Training via Spectral Normalization](https://arxiv.org/abs/1811.07469).
