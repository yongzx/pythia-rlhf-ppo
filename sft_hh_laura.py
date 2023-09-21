#### commands to run ####
# accelerate launch --config_file accelerate_config.yaml sft_hh.py
#########################
import json
import sys

from datasets import load_dataset
from ppo_hh import create_reward_fn

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    SFTConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

warmup_steps = 150 # https://github.com/lomahony/direct-preference-optimization/blob/main/config/config.yaml
default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=1,
        total_steps=100000000000000, # so the training steps = epochs * len(dataset)
        batch_size=32,
        checkpoint_interval=1000,
        eval_interval=1000,
        pipeline="PromptPipeline",
        trainer="AccelerateSFTTrainer",
        checkpoint_dir="checkpoints/sft_hh_laura/pythia-70m",
        group_name = "EleutherAI/pythia-70m",
        entity_name = 'eleutherai',
        project_name = 'pythia-rlhf',
    ),
    model=ModelConfig(model_path="EleutherAI/pythia-70m", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/pythia-160m", truncation_side="left"),
    # customize the trlx/utils/__init__.py
    optimizer=OptimizerConfig(name="rmsprop"), # use default hp
    scheduler=SchedulerConfig(name="lambda_lr", kwargs=dict(lr_lambda=lambda step: min(1.0, (step + 1) / (warmup_steps + 1)))),
    method=SFTConfig(
        name="sftconfig",
        gen_kwargs=dict(max_new_tokens=128, top_k=20, top_p=1.0, do_sample=True),
    ),
)


def preprocess(sample):
    sample["chosen_sample"] = sample["prompt"] + sample["chosen"]
    return sample


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("Dahoas/full-hh-rlhf").map(preprocess)
    trlx.train(
        config=config,
        samples=dataset["train"]["chosen_sample"],
        eval_prompts=dataset["test"]["prompt"][:40],
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)