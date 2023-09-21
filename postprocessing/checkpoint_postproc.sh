# add pytorch_model.bin, config.json, and tokenizer's related config files
# to the checkpoint directory

MODEL_DIR="/mnt/ssd-1/pythia-rlhf/checkpoints/sft_hh_laura/pythia-70m/"
MODEL_NAME="EleutherAI/pythia-70m"
TOK_NAME=$MODEL_NAME

source /mnt/ssd-1/pythia-rlhf/venv/bin/activate
for CKPT in $MODEL_DIR/*/ ; do
    echo ">>>> Processing $CKPT" # absolute path

    python3.9 "${CKPT}/zero_to_fp32.py" "${CKPT}" "${CKPT}/pytorch_model.bin" 
    python3.9 "/mnt/ssd-1/pythia-rlhf/postprocessing/generate_config_and_tokenizer.py" \
        "${CKPT}" \
        $MODEL_NAME \
        --tokenizer_name=$TOK_NAME
done
