# NOTES

```
accelerate launch --config_file accelerate_config.yaml sft_hh.py &> output_ppo_70m.out 

## in checkpoint dir
python3.9 zero_to_fp32.py . pytorch_model.bin


accelerate launch --config_file accelerate_config.yaml ppo_hh.py &> output_ppo_70m.out
```

### HP Sweep
