# ml-group-02 — Reproducing & Critiquing L2P

Term project, Introduction to Machine Learning (HCMUS).
Paper: Z. Wang et al., "Learning to Prompt for Continual Learning," *CVPR*, 2022. [[CVF]](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.html)

- W&B project (public): `TODO link`
- Plan: [ROADMAP.md](ROADMAP.md) · Spec: [docs/l2p_spec.md](docs/l2p_spec.md) · Experiments: [docs/experiments.md](docs/experiments.md)

## Team

| Member | Student ID | Role |
|---|---|---|
| TODO | | |

## Repository layout

```
src/            # own implementation of L2P (core written from the paper)
configs/        # YAML configs, one per experiment
notebooks/      # thin Kaggle wrappers that call `python -m src...`
docs/           # spec, experiment plan, critique notes
Research/       # papers (PDF)
```

## Experiment map

Every number in the deck/poster links to a W&B run listed here.

| ID | Script / notebook | W&B run | Result (one line) |
|---|---|---|---|
| R1a | TODO | TODO | TODO |

## Third-party code & libraries

The L2P core (prompt pool, query–key selection, surrogate loss, training loop) is implemented by the team from the paper's description. The official code is used only for reading and output comparison.

| Library | Used for |
|---|---|
| PyTorch | training |
| timm | pretrained ViT-B/16 backbone |
| torchvision / HF datasets | CIFAR-100 loading |
| wandb | experiment tracking |

## Reproduce

See [docs/kaggle_setup.md](docs/kaggle_setup.md).
