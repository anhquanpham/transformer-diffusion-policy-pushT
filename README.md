# Transformer-Based Visuomotor Diffusion Imitation Learning for Robot Manipulation

More details in project page: https://anhquanpham.github.io/projects/transformer-diffusion-immitation 

Chi et al. (2023) introduced the Diffusion Policy framework, which extends Denoising Diffusion Probabilistic Models (DDPMs) to visuomotor policy learning. This framework models actions as sequential outputs, refining noisy samples into task-specific actions using a learned score function. While innovative, the original framework has limitations, including difficulty modeling long-term dependencies across sequential observations and actions.

To address these challenges, this project enhances the Diffusion Policy framework by integrating transformer-based observation and action encoders. The self-attention mechanism of transformers enables the modeling of both short- and long-term temporal dependencies, balancing temporal consistency (coherent actions) and responsiveness (quick adaptation to observations). Additionally, the enhanced architecture includes a custom denoising process that integrates outputs from the encoders for robust and consistent action generation.




