# Generative Modeling of 3D Chair Point Clouds

This repository contains an exploratory implementation of a
PointNet-style variational autoencoder for generating 3D chair
geometry represented as point clouds.

The experimental pipeline includes:

- normalization of ModelNet chair meshes;
- surface sampling into point clouds;
- PointNet-style encoding;
- variational latent-space learning;
- generation of new point clouds;
- preliminary evaluation using Chamfer Distance;
- Poisson surface reconstruction.

> **Project status:** experimental prototype. The model captures
> coarse chair-like structure but poorly preserves thin components
> such as legs. Surface reconstruction produces substantial artifacts
> and is not suitable for engineering use.

<img width="1087" height="723" alt="generated_point_clouds" src="https://github.com/user-attachments/assets/4b7ded5d-3d33-4889-bfe7-13377f960eab" />


*Random samples decoded from the latent space. The generated point
clouds capture coarse seat-and-back structure, but point density is
uneven and thin elements are poorly represented.*

## Original experiment

- Dataset: ModelNet, chair category
- Training objects: approximately 890
- Stored points per object: 16,384
- Encoder input points: 4,096
- Latent dimension: 256
- Optimizer: Adam
- Learning rate: `1e-4`
- KL coefficient: `1e-4`
- Training duration: 25 epochs
- Selected checkpoint: epoch 23

Training continued for 25 epochs, but the checkpoint from epoch 23
was selected because validation Chamfer Distance deteriorated during
the final two epochs.

## Preliminary evaluation

Ten generated point clouds were compared with the held-out validation
subset.

| Metric | Value |
|---|---:|
| Mean nearest-reference Chamfer-L2 | 0.0375 |
| Minimum nearest-reference Chamfer-L2 | 0.0306 |
| Maximum nearest-reference Chamfer-L2 | 0.0431 |
| Mean pairwise Chamfer-L2 | 0.0306 |

The reported metric uses non-squared Euclidean distances. These
measurements provide a preliminary estimate of geometric similarity
but do not establish perceptual quality, novelty or correct structural
composition.

## Surface reconstruction

Generated point clouds were converted into polygonal meshes using
Poisson Surface Reconstruction.

<img width="1111" height="612" alt="poisson_reconstruction_failures" src="https://github.com/user-attachments/assets/78a5e32a-e0e0-4383-90c7-3c27cbd8ff67" />


*Poisson reconstruction preserves some coarse seat and back regions
but loses chair legs and introduces severe surface artifacts.*

The reconstruction results were unsatisfactory because the generated
point clouds have uneven density, especially around thin structural
elements.

## Limitations

- small dataset relative to model capacity;
- heavily overparameterized fully connected decoder;
- uneven generated point density;
- limited evaluation protocol;
- no independent final test set;
- loss of thin components during surface reconstruction;
- full historical training logs and trained weights are not included.

## Future work

- compare the VAE with point-cloud diffusion models;
- replace the fully connected decoder with a more efficient
  coarse-to-fine or folding-based architecture;
- add density-aware and part-aware objectives;
- evaluate fidelity and coverage on a separate test split;
- explore implicit-surface and mesh-native decoders.
