# HyperGAN on MNIST: generating neural-network weights with a GAN

An exploratory personal project. **Can a GAN learn to generate the *weights* of working classifiers, and can such a generator be adapted to a new class from only a few examples?**

The idea comes from the weight-space learning literature: HyperNetworks (Ha et al., 2016), HyperGAN (Ratzlaff & Fuxin, 2019), G.pt (Peebles et al., 2022) and Neural Network Diffusion (Wang et al., 2024).

## Setup

| Phase | What happens |
|---|---|
| **0: Model zoo** | Train 250 small MLPs (196 → 32 → 10, MNIST downsampled to 14×14) on digits **0–8** only. Each network has 6,634 parameters. |
| **1: GAN** | Train a GAN whose samples are 6,634-dimensional weight vectors. Measure the accuracy of the networks it generates. |
| **2: Few-shot** | Fine-tune the generator with only **K = 20** examples of the unseen digit **9** (plus a small replay set of digits 0–8). Check whether the generated networks now recognise all 10 digits. |

## Three tricks that make it work

1. **Shared initialisation against permutation symmetry.** Permuting hidden neurons leaves an MLP's function unchanged but scatters its weight vector. If every zoo member is trained from the *same* initial weights, the networks stay aligned in weight space and the distribution collapses into a single basin.
2. **Per-dimension normalisation.** The GAN works in a weight space standardised by the zoo's mean and standard deviation.
3. **Differentiable task loss.** Generated vectors are plugged into a skeleton network with `torch.func.functional_call` and evaluated on real MNIST batches. The resulting cross-entropy is back-propagated into the generator alongside the adversarial loss: `L_G = L_adv + λ·L_task`.

## Results (single run, seed 0, GPU)

| | Accuracy (digits 0–8) | Accuracy (10 classes) | Recall on digit 9 |
|---|---|---|---|
| Model zoo (mean of 250) | 90.6% | n/a | n/a |
| Phase-1 generator (16 sampled nets) | 90.9% ± 0.1 | 81.7% | 0.0% |
| Phase-2 generator after 20-shot fine-tuning | n/a | 89.5% ± 0.1 | 80.1% |
| Phase-2 ensemble of 16 generated nets | n/a | 89.6% | 80.5% |
| Baseline: one zoo net fine-tuned on the same data | n/a | 89.4% | 77.1% |

## Honest caveats

- **Low diversity / partial mode collapse.** Generated networks disagree on only 0.56% of test predictions. Their mean pairwise distance in normalised weight space is 22.9, compared with 114.4 for the zoo. The task loss pulls samples towards one strong solution, so the "free ensemble" adds little.
- **The baseline is close.** The few-shot gain over plain fine-tuning (+3 points recall on digit 9) comes from a single seed and is not statistically established.
- **Toy scale.** A flat MLP generator does not scale to millions of parameters. Real systems generate layer-wise or in a latent space, often with diffusion rather than GANs.
- **The alignment trick is a shortcut.** Zoos trained from different initialisations would need permutation-equivariant architectures or alignment methods such as Git Re-Basin.

## Run it

```bash
pip install torch torchvision numpy matplotlib tqdm jupyterlab
jupyter lab hypergan_mnist.ipynb
```

The notebook takes about 3–5 minutes on a GPU and 10–25 minutes on a CPU. Its explanatory text and plot labels are in Turkish; code and results are language-independent.

*The code was written with the help of an AI coding assistant. The experiment design, runs and analysis are my own.*
