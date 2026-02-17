# src/agent.py
from typing import Dict, Any
import os
import datetime
import torch

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO


def _build_tb_dir(paths_config: Dict[str, Any], traincfg: Dict[str, Any]) -> str:
    folder_name = traincfg.get("folder_name")
    run_name = traincfg.get("run_name") or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = folder_name if folder_name else run_name  # prefer folder_name for directory
    tb_root  = paths_config["log_dir"]                   # from YAML: paths.log_dir
    tb_dir   = os.path.join(tb_root, dir_name)
    os.makedirs(tb_dir, exist_ok=True)
    return tb_dir


def _select_device(agent_config: Dict[str, Any]) -> str:
    wanted_device = agent_config.get("device", "auto")  # allow "cuda", "cpu", or "auto"
    if wanted_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return wanted_device


def _common_kwargs(agent_cfg: Dict[str, Any], tb_dir: str, device: str, verbose: int) -> Dict[str, Any]:
    return dict(
        verbose=verbose,
        tensorboard_log=tb_dir,
        device=device,
        learning_rate=agent_cfg.get("learning_rate", 1e-4),
        n_steps=agent_cfg.get("n_steps", 1024),
        batch_size=agent_cfg.get("batch_size", 64),
        n_epochs=agent_cfg.get("n_epochs", 10),
        ent_coef=agent_cfg.get("ent_coef", 0.0),
        gamma=agent_cfg.get("gamma", 0.995),
        clip_range=agent_cfg.get("clip_range", 0.2),
        gae_lambda=agent_cfg.get("gae_lambda", 0.95),
        vf_coef=agent_cfg.get("vf_coef", 0.5),
        max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
        target_kl=agent_cfg.get("target_kl", None),
    )


def create_agent(env, config):
    """
    Creates a PPO or RecurrentPPO agent based on the provided configuration.

    Selection rule (no extra flags needed):
      - policy == "MlpLstmPolicy" -> RecurrentPPO (LSTM)
      - policy == "MlpPolicy"     -> PPO (feed-forward)
    """
    agent_config = config["agent"]
    paths_config = config["paths"]
    traincfg     = config["training"]

    policy = agent_config.get("policy", "MlpLstmPolicy")
    use_recurrent = policy.lower() == "mlplstmpolicy"

    tb_dir = _build_tb_dir(paths_config, traincfg)
    device = _select_device(agent_config)
    print(f"→ Torch CUDA available? {torch.cuda.is_available()} | Using device: {device}")
    print(f"-> Creating {'RecurrentPPO' if use_recurrent else 'PPO'} agent with policy '{policy}'...")

    verbose = traincfg.get("verbose", 1)
    common_kwargs = _common_kwargs(agent_config, tb_dir, device, verbose)

    if use_recurrent:
        # LSTM-specific configuration (from your YAML)
        shared_lstm = agent_config.get("shared_lstm", True)
        enable_critic_lstm = agent_config.get("enable_critic_lstm", False)

        # Safety checks (only relevant for recurrent)
        if shared_lstm and enable_critic_lstm:
            raise ValueError("Choose either shared_lstm=True OR enable_critic_lstm=True, not both.")

        policy_kwargs = dict(
            lstm_hidden_size=agent_config.get("lstm_hidden_size", 128),
            n_lstm_layers=agent_config.get("n_lstm_layers", 1),
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
        )

        model = RecurrentPPO(
            policy,
            env,
            **common_kwargs,
            policy_kwargs=policy_kwargs,
        )
        print(f"→ LSTM mode: {'shared' if shared_lstm else ('separate' if enable_critic_lstm else 'no-critic-lstm')}")
    else:
        # PPO ignores LSTM-specific fields silently
        policy_kwargs = agent_config.get("policy_kwargs", {})
        model = PPO(
            policy,
            env,
            **common_kwargs,
            policy_kwargs=policy_kwargs,
        )

    print("✓ Agent created.")
    return model


def load_agent(path: str, env):
    """
    Loads a pre-trained PPO or RecurrentPPO agent from a file.
    Tries PPO first, then RecurrentPPO.
    """
    print(f"-> Loading agent from: {path}")
    try:
        model = PPO.load(path, env=env)
        print("✓ Loaded PPO checkpoint.")
        return model
    except Exception as e_ppo:
        try:
            model = RecurrentPPO.load(path, env=env)
            print("✓ Loaded RecurrentPPO checkpoint.")
            return model
        except Exception as e_rppo:
            print(f"ERROR: Failed to load model. PPO error: {e_ppo} | RecurrentPPO error: {e_rppo}")
            return None
