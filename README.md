# genDOPA

Generative model for designing catechol-based adhesive molecules using Variational Autoencoder (VAE).

## Overview
`genDOPA` implements VAE and conditional VAE (cVAE) models to generate catechol-derived adhesive molecules with improved adsorption energy on graphite.  
Molecular data are represented with (group)SELFIES or SMILES, and the model learns latent representations that capture chemical and physical features.

**Status:** *In progress* — currently building data pipeline, neural network module, and training configuration.

## Usage

```bash
git clone https://github.com/SeyongChoi/genDOPA_toy.git
cd genDOPA_toy
pip install -r requirements.txt

# Train model
python main.py --config config/train_cvae.yaml

# Generate new molecules
python main.py --config config/generate.yaml
