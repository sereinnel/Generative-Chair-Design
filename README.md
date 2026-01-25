# Generative-Chair-Design
Generative 3D Chair Design from Point Clouds
A research-driven pipeline that generates novel, plausible 3D chair designs using a PointNet-based Variational Autoencoder — trained directly on raw, fragmented meshes from ModelNet40.

This project demonstrates how to build a full generative design workflow for 3D objects without requiring clean, watertight meshes. Instead, it leverages point clouds — a robust representation for real-world, incomplete CAD data — to:

Train a VAE on non-manifold, multi-part chair models
Generate diverse, realistic chair geometries via latent-space sampling
Quantitatively evaluate quality with Chamfer Distance
Reconstruct surfaces into .obj meshes (with honest limitations documented)
Perfect for showcasing skills in:

3D deep learning (PointNet, VAEs)
Geometric data processing (Open3D, mesh normalization, point cloud sampling)
Reproducible ML pipelines (config-driven, modular scripts)
Critical analysis of reconstruction trade-offs
Most generative 3D models require perfect, closed meshes — but real-world CAD data is messy.  
By using **point clouds** (unordered sets of 3D points), we bypass the need for topological consistency.  
The **VAE** learns a compact latent space where interpolation and sampling produce plausible new designs — all without ever seeing a complete mesh during training.

💡 Why it matters: Most generative 3D methods require perfect input data — but real-world CAD is messy. This project proves you can work with imperfect, open-source datasets and still produce meaningful results.
