# src/train.py
from . import tf_config  # keep your TF quiet config

import os, time, math, datetime
import yaml
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from . import utils, agent
from .callbacks import SecLoggingCallback, ResourceUsageCallback
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    start_time = time.time()

    # --- 1) Load config & make envs ---
    config = utils.load_config()
    precomputed_data = utils.load_and_preprocess_data(config)
    train_env, eval_env = utils.create_vectorized_envs(config, precomputed_data)

    train_env = VecMonitor(train_env)
    eval_env  = VecMonitor(eval_env)

    setup_time = time.time() - start_time
    print(f"\n✓ Setup completed in {setup_time:.2f} s")

    paths   = config['paths']
    traincf = config['training']

    # --- 2) Make a unique experiment folder (exp_dir) ---
    folder_name = traincf.get("folder_name")  # use folder_name if provided
    run_name = traincf.get("run_name") or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = folder_name if folder_name else run_name  # prefer folder_name for directory
    base = paths['best_model_save_path'].rstrip("/")
    exp_dir = os.path.join(paths['best_model_save_path'].rstrip("/"), dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"→ Experiment dir: {exp_dir}")
    print(f"→ Run name: {run_name}")

    # Save a snapshot of the exact run configuration for reproducibility
    config_snapshot_path = os.path.join(exp_dir, "config.yaml")
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"✓ Run config snapshot saved to: {config_snapshot_path}")

    # --- 3) Create agent ---
    model = agent.create_agent(train_env, config)

    # --- 4) Callbacks (all point to exp_dir) ---
    eval_freq_base = traincf['eval_freq_denom']
    eval_freq = max(math.ceil(eval_freq_base / train_env.num_envs), 500)
    print(f"→ Eval every {eval_freq} steps; n_eval_episodes={traincf['n_eval_episodes']}")

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=exp_dir,
        log_path=exp_dir,
        eval_freq=eval_freq,
        n_eval_episodes=traincf['n_eval_episodes'],
        deterministic=traincf['deterministic_eval'],
        render=False,
    )

    sec_cb = SecLoggingCallback(
        log_dir=exp_dir,
        tb_prefix="train",
        verbose=traincf.get("verbose", 1),
    )
    res_cb = ResourceUsageCallback(    
        log_dir=exp_dir,
        tb_prefix="sys",
        log_every_n_steps=traincf.get('log_every_n_steps', 1000),
        verbose=traincf.get("verbose", 1),
    )

    ckpt_cb = CheckpointCallback(
        save_freq=traincf.get('checkpoint_freq', 50_000),
        save_path=exp_dir,
        name_prefix="ckpt",
    )

    # --- 5) Learn ---
    model.learn(
        total_timesteps=traincf['total_timesteps'],
        callback=[sec_cb, eval_cb, ckpt_cb,res_cb],
        progress_bar=traincf.get('progress_bar', True),
    )

    tr_time = time.time() - start_time - setup_time
    print(f"\n✓ Training completed in {tr_time:.2f} s")

    # --- 6) Save final artifacts under the same exp_dir ---
    final_model_path = os.path.join(exp_dir, "final_model")
    vecnorm_path     = os.path.join(exp_dir, "vec_normalize_stats.pkl")
    model.save(final_model_path)
    train_env.save(vecnorm_path)
    print(f"\033[92m✓ Final model saved to: {final_model_path}.zip\033[0m")
    print(f"\033[92m✓ VecNormalize stats saved to: {vecnorm_path}\033[0m")

    total_time = time.time() - start_time
    print(f"\nTotal script execution time: {total_time:.2f} s")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Configured memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    train()
