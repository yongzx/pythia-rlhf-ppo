`checkpoint_postproc.sh` adds the pytorch_model.bin, config.json, and tokenizer's related config files to the checkpoint directory after training with Deepspeed. The files are necessary for loading with HF transformers library.

`upload_checkpoints_hf.sh` is used to upload the model's checkpoints to HF Hub.