# genDOPA

Generative model for designing catechol-based adhesive molecules using Variational Autoencoder (VAE).

## Overview

`genDOPA` implements VAE and conditional VAE (cVAE) models to generate catechol-derived adhesive molecules with improved adsorption energy on graphite.  
Molecular data are represented with (Group)SELFIES or SMILES, and the model learns latent representations that capture chemical and physical features.

-----
**Status:** *In progress* 
- [ ] Expand the dataset from ZINC  
- [x] Build and verify data preprocessing to express in GroupSELFEIS
- [x] Implement VAE modules 
- [ ] Implement cVAE modules   
- [ ] Implement train module and finalize training configuration (YAML)
- [ ] Add molecular visualization and property evaluation tools  
- [ ] Conduct initial generation and reconstruction tests  
-----

## Usage

```bash
git clone https://github.com/SeyongChoi/genDOPA_toy.git
cd genDOPA_toy
pip install -r requirements.txt

# Train model
python main.py --config config/train_cvae.yaml

# Generate new molecules
python main.py --config config/generate.yaml
