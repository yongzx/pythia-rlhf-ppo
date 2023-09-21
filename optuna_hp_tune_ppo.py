import optuna
import subprocess
import json
import shutil
MAX_LAYERS_UNFROZEN = 8


def objective(trial):
    config = {
        "model.num_layers_unfrozen": trial.suggest_int("layers_unfrozen", 1, MAX_LAYERS_UNFROZEN),
        "method.target": trial.suggest_float("kl", 2, 8),
        "method.ppo_epochs": trial.suggest_int("epochs_per_batch", 2, 8),
        "train.total_steps": 100,
        "train.checkpoint_dir": f"checkpoints/ppo_hh/optuna_trial_{trial.number}/"
    }
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-5, log = True)
    run = subprocess.Popen([
        "accelerate", "launch", 
        "--config_file", "accelerate_config.yaml", "ppo_hh_hpsweep.py", 
        json.dumps(config), str(learning_rate)],
        stdout = subprocess.PIPE,
        bufsize = 1,
        universal_newlines = True,
    )

    for line in run.stdout:
        print(line)
    
    run.wait()
    avg_score = float(line)

    print("Score Of Trial: ", avg_score)

    shutil.rmtree(f"checkpoints/ppo_hh/optuna_trial_{trial.number}")
    return avg_score


study_name = '2.8b-hparams'
study = optuna.create_study(
    study_name = study_name, 
    direction = 'maximize', 
    storage = 'sqlite:///example.db',
    load_if_exists = True
)  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.