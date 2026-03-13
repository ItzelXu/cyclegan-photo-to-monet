# CycleGAN: Photo-to-Monet Style Transfer

An implementation of **CycleGAN** for unpaired image-to-image translation, converting real photographs into Claude Monet impressionist paintings — and back.

> Kaggle Competition: *"I'm Something of a Painter Myself"*

## Background

Traditional style transfer requires paired training data (photograph A and its Monet equivalent). CycleGAN's breakthrough is that it learns the mapping from **unpaired** datasets — just a collection of photos and a separate collection of Monet paintings, with no correspondence between them.

The key insight: instead of directly mapping A→B, enforce a **cycle consistency constraint**: A → B → A must reconstruct the original A.

## Architecture

```
Domain X (Photos)          Domain Y (Monet Paintings)
      │                              │
      │      Generator G_XY          │
      │ ─────────────────────► Fake Y│
      │                              │
      │      Generator G_YX          │
      │ ◄───────────────────── Fake X│
      │                              │
      │    Cycle Consistency:        │
      │    X → G_XY → G_YX ≈ X      │
      │    Y → G_YX → G_XY ≈ Y      │
      │                              │
 Discriminator D_X            Discriminator D_Y
 (Real / Fake X?)             (Real / Fake Y?)
```

### Generator Architecture
```
Input (3, H, W)
  → ConvLayer(3→64, k=7, s=1)    ← Instance Norm + ReLU
  → ConvLayer(64→128, k=3, s=2)  ← Downsample
  → ConvLayer(128→256, k=3, s=2) ← Downsample
  → ResBlock × 9                  ← Identity residuals (preserve structure)
  → Upsample(256→128)             ← Transposed Conv + IN + ReLU
  → Upsample(128→64)              ← Transposed Conv + IN + ReLU
  → ConvLayer(64→3, k=7)         ← Tanh activation → [-1, 1]
```

### Discriminator: PatchGAN
70×70 receptive field — classifies overlapping image patches as real/fake rather than the entire image, enabling finer texture discrimination.

## Loss Functions

```python
# Adversarial loss: fool the discriminator
L_GAN = -E[log D(G(x))]

# Cycle-consistency loss (structural preservation)
L_cycle = ‖G_YX(G_XY(x)) - x‖₁ · λ        # λ = 10

# Identity loss (color preservation)
L_idt = ‖G_XY(y) - y‖₁ · λ · idt_coef     # λ=10, idt_coef=0.5

L_total = L_GAN + L_cycle + L_idt
```

**Why λ=10 for cycle loss?** Forces the generator to "remember" the original scene structure (mountains, trees, sky) while adding painterly brushstrokes. Without this high weight, mode collapse occurs — generators find shortcuts (e.g., uniform color washes).

**Why identity loss?** Prevents the generator from inverting colors when given an already-Monet-style input. Stabilizes training and preserves color fidelity.

## Key Engineering: Gradient Isolation

A critical implementation detail: generator and discriminator must be updated **alternately**, not simultaneously.

```python
def update_req_grad(models, requires_grad: bool):
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad

# Train Generator: freeze discriminators
update_req_grad([disc_M, disc_P], False)
gen_loss.backward()
adam_gen.step()

# Train Discriminator: unfreeze
update_req_grad([disc_M, disc_P], True)
disc_loss.backward()
adam_desc.step()
```

This prevents gradient cross-contamination and mirrors game-theory alternating updates — the discriminator serves as a fixed judge while the generator improves, then vice versa.

## Challenges & Solutions

| Problem | Cause | Solution |
|---|---|---|
| Mode collapse | Generator found a shortcut (all-blue tint) | High λ cycle loss + identity loss |
| Structure loss | Trees/skylines distorted | λ=10 forces geometric reconstruction fidelity |
| Color inversion | White sky → dark artifacts | Identity loss prevents untargeted color changes |
| Training instability | Simultaneous G+D updates | `requires_grad` isolation per update step |

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Framework | PyTorch |
| Generator | ResNet-9 with 9 residual blocks |
| Discriminator | PatchGAN (70×70 receptive field) |
| Normalization | Instance Normalization |
| Optimizer | Adam (lr=2e-4, β₁=0.5) |
| Loss | Adversarial + Cycle-Consistency + Identity |

## How to Run

```bash
pip install torch torchvision matplotlib pillow

# Review architecture and training
# See: CycleGan.pdf for full code walkthrough
```

## Repository Structure

```
cycleGan/
├── README.md
├── CHANGELOG.md
├── CycleGan.pdf                         ← Full code + architecture documentation
├── fast_style_transfer.pdf              ← Background reading: neural style transfer
├── painter_myself_competition_notes.docx ← Kaggle competition notes
├── report.docx                          ← Project report with output samples
└── gemini_project_analysis.txt          ← AI-assisted architecture deep-dive
```

---

*Academic/Kaggle Project · Python · PyTorch · Generative Adversarial Networks · Image-to-Image Translation*
