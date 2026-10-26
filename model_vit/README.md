# ViT-Transformer model

This folder contains the model, training and inference sources and configuration
adapted from `dvankien123/image-captioning` main (fetched at commit `7793d85`). It is
kept separate from the older ResNet-LSTM files in `model/` and the existing
top-level `src/` directory.

The backend loads `checkpoints/best_model.pth` by default. Trained checkpoints
are kept out of Git because of their size; place weights produced by this
ViT-Transformer model there before starting caption inference. The ResNet-LSTM
checkpoint in `../model/checkpoints/` cannot be loaded by this model.

Dataset and checkpoint paths for the model workflow are in `config.yaml` and
are relative to this folder. The backend can use a different compatible
checkpoint by setting `MODEL_CHECKPOINT` in the repository root `.env` file.
