# callbacks.py
import os, pandas as pd, time
from stable_baselines3.common.callbacks import BaseCallback


import psutil

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


class SecLoggingCallback(BaseCallback):
    """
    Logs per-episode aggregates to CSV and TensorBoard.
    Works with VecEnvs; expects your env to populate 'info' keys you already added.
    """
    def __init__(self, log_dir: str, tb_prefix: str = "train", verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb_prefix = tb_prefix
        self.rows = []
        self._init_stats = False

    def _init_per_env(self):
        n = getattr(self.training_env, "num_envs", 1)
        self.stats = [dict(rew=0., len=0, sec=0., power=0., perf=0.,
                           qos_viol=0, switches=0, scales=0, time_oai=0, k_sum=0)
                      for _ in range(n)]
        self._init_stats = True

    def _on_training_start(self):
        self._init_per_env()

    def _on_step(self) -> bool:
        if not self._init_stats:
            self._init_per_env()
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        n = len(infos) if isinstance(infos, (list, tuple)) else 1
        for i in range(n):
            info = infos[i] if n > 1 else infos
            st = self.stats[i]
            st["len"] += 1
            if rewards is not None: st["rew"] += float(rewards[i])
            st["sec"]   += float(info.get("sec") or 0.0)
            st["power"] += float(info.get("power") or 0.0)
            st["perf"]  += float(info.get("performance") or 0.0)
            st["qos_viol"] += 1 if (info.get("qos_penalty") or 0.0) > 0 else 0
            st["switches"] += 1 if (info.get("type_switch_penalty") or 0.0) > 0 else 0
            st["scales"]   += 1 if (info.get("scale_penalty") or 0.0) != 0 else 0
            chosen = str(info.get("chosen_upf", ""))
            if "OAI" in chosen:
                st["time_oai"] += 1
                try: st["k_sum"] += int(chosen.split("x")[0])
                except: pass

            if dones is not None and dones[i]:
                L = max(st["len"], 1)
                row = dict(
                    timesteps=self.num_timesteps,
                    ep_return=st["rew"], ep_len=st["len"],
                    sec_mean=st["sec"]/L, power_mean=st["power"]/L, perf_mean=st["perf"]/L,
                    qos_violation_rate=st["qos_viol"]/L,
                    switches=st["switches"], scale_events=st["scales"],
                    frac_time_oai=st["time_oai"]/L,
                    mean_k=(st["k_sum"]/max(st["time_oai"],1)) if st["time_oai"]>0 else 0.0,
                )
                self.rows.append(row)
                for k,v in row.items():
                    self.logger.record(f"{self.tb_prefix}/{k}", v)
                self.stats[i] = dict(rew=0., len=0, sec=0., power=0., perf=0.,
                                     qos_viol=0, switches=0, scales=0, time_oai=0, k_sum=0)
        return True

    def _on_training_end(self):
        if self.rows:
            pd.DataFrame(self.rows).to_csv(os.path.join(self.log_dir, "train_episodes.csv"), index=False)


class ResourceUsageCallback(BaseCallback):
    """
    Enhanced callback that logs system-level resource usage (CPU, RAM, GPU util/power) during training.
    - Writes a CSV time-series: resource_usage.csv
    - Logs metrics to TensorBoard every `log_every_n_steps` steps
    - Tracks total energy consumption during training
    """
    def __init__(self, log_dir: str, tb_prefix: str = "sys",
                 log_every_n_steps: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb_prefix = tb_prefix
        self.log_every_n_steps = log_every_n_steps
        self.rows = []
        self._last_log_step = 0
        self._nvml_inited = False
        self._gpu_handle = None
        self._last_measurement_time = None
        self._cumulative_energy_wh = 0.0  # Track total energy consumed
        self._gpu_capabilities = {'power': False, 'memory': False, 'utilization': False, 'temperature': False}

    def _init_nvml(self):
        if _HAS_NVML and not self._nvml_inited:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_inited = True
                if self.verbose:
                    gpu_name = pynvml.nvmlDeviceGetName(self._gpu_handle)
                    print(f"ResourceUsageCallback: NVML initialized for {gpu_name}")
                
                # Detect GPU capabilities
                try:
                    pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                    self._gpu_capabilities['power'] = True
                except:
                    pass
                try:
                    pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    self._gpu_capabilities['memory'] = True
                except:
                    pass
                try:
                    pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    self._gpu_capabilities['utilization'] = True
                except:
                    pass
                try:
                    pynvml.nvmlDeviceGetTemperature(self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                    self._gpu_capabilities['temperature'] = True
                except:
                    pass
                    
            except Exception as e:
                if self.verbose:
                    print(f"ResourceUsageCallback: NVML init failed: {e}")
                self._nvml_inited = False

    def _on_training_start(self) -> None:
        self._init_nvml()
        self._start_time = time.time()
        self._last_measurement_time = time.time()

    def _capture_metrics(self):
        current_time = time.time()
        ts = current_time - self._start_time

        cpu_pct = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024**3)  # GB
        ram_pct  = ram.percent

        gpu_power_w = None
        gpu_mem_gb  = None
        gpu_util    = None
        gpu_temp_c  = None

        if self._nvml_inited:
            # Only query supported metrics
            if self._gpu_capabilities.get('power'):
                try:
                    p = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle) / 1000.0  # watts
                    gpu_power_w = p
                    
                    # Calculate energy consumption since last measurement
                    if self._last_measurement_time is not None:
                        time_delta_hours = (current_time - self._last_measurement_time) / 3600.0
                        energy_delta_wh = gpu_power_w * time_delta_hours
                        self._cumulative_energy_wh += energy_delta_wh
                except Exception:
                    pass
            
            if self._gpu_capabilities.get('memory'):
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    gpu_mem_gb = mem_info.used / (1024**3)
                except Exception:
                    pass
            
            if self._gpu_capabilities.get('utilization'):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    gpu_util = util.gpu
                except Exception:
                    pass
            
            if self._gpu_capabilities.get('temperature'):
                try:
                    gpu_temp_c = pynvml.nvmlDeviceGetTemperature(self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    pass

        self._last_measurement_time = current_time

        row = dict(
            timesteps=self.num_timesteps,
            elapsed_s=ts,
            cpu_pct=cpu_pct,
            cpu_freq_mhz=cpu_freq.current if cpu_freq else None,
            ram_gb=ram_used,
            ram_pct=ram_pct,
            gpu_power_w=gpu_power_w,
            gpu_mem_gb=gpu_mem_gb,
            gpu_util_pct=gpu_util,
            gpu_temp_c=gpu_temp_c,
            cumulative_energy_wh=self._cumulative_energy_wh,
        )
        self.rows.append(row)

        # Log to TensorBoard
        self.logger.record(f"{self.tb_prefix}/cpu_pct", cpu_pct)
        self.logger.record(f"{self.tb_prefix}/ram_gb", ram_used)
        if gpu_power_w is not None:
            self.logger.record(f"{self.tb_prefix}/gpu_power_w", gpu_power_w)
            self.logger.record(f"{self.tb_prefix}/cumulative_energy_wh", self._cumulative_energy_wh)
            self.logger.record(f"{self.tb_prefix}/cumulative_energy_kwh", self._cumulative_energy_wh / 1000)
        if gpu_util is not None:
            self.logger.record(f"{self.tb_prefix}/gpu_util_pct", gpu_util)
        if gpu_temp_c is not None:
            self.logger.record(f"{self.tb_prefix}/gpu_temp_c", gpu_temp_c)

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_log_step) >= self.log_every_n_steps:
            self._capture_metrics()
            self._last_log_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.rows:
            out_path = os.path.join(self.log_dir, "resource_usage.csv")
            pd.DataFrame(self.rows).to_csv(out_path, index=False)
            if self.verbose:
                print(f"ResourceUsageCallback: wrote {len(self.rows)} rows to {out_path}")
            
            # Save training resource summary
            summary = {
                'total_duration_s': self.rows[-1]['elapsed_s'] if self.rows else 0,
                'total_timesteps': self.num_timesteps,
                'num_measurements': len(self.rows),
                'cumulative_energy_wh': self._cumulative_energy_wh,
                'cumulative_energy_kwh': self._cumulative_energy_wh / 1000,
            }
            
            # Add statistics if available
            df = pd.DataFrame(self.rows)
            if 'cpu_pct' in df.columns:
                summary['cpu_pct_mean'] = float(df['cpu_pct'].mean())
                summary['cpu_pct_max'] = float(df['cpu_pct'].max())
            if 'ram_gb' in df.columns:
                summary['ram_gb_mean'] = float(df['ram_gb'].mean())
                summary['ram_gb_max'] = float(df['ram_gb'].max())
            if 'gpu_power_w' in df.columns and df['gpu_power_w'].notna().any():
                summary['gpu_power_w_mean'] = float(df['gpu_power_w'].mean())
                summary['gpu_power_w_max'] = float(df['gpu_power_w'].max())
            if 'gpu_util_pct' in df.columns and df['gpu_util_pct'].notna().any():
                summary['gpu_util_pct_mean'] = float(df['gpu_util_pct'].mean())
                summary['gpu_util_pct_max'] = float(df['gpu_util_pct'].max())
            if 'gpu_temp_c' in df.columns and df['gpu_temp_c'].notna().any():
                summary['gpu_temp_c_mean'] = float(df['gpu_temp_c'].mean())
                summary['gpu_temp_c_max'] = float(df['gpu_temp_c'].max())
            
            # Save summary JSON
            import json
            summary_path = os.path.join(self.log_dir, "training_resource_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n{'='*60}")
            print("  TRAINING RESOURCE USAGE SUMMARY")
            print('='*60)
            print(f"Total Duration:     {summary['total_duration_s']:.2f} seconds")
            print(f"Total Timesteps:    {summary['total_timesteps']}")
            if 'cpu_pct_mean' in summary:
                print(f"CPU Usage (mean):   {summary['cpu_pct_mean']:.1f}%")
            if 'ram_gb_mean' in summary:
                print(f"RAM Usage (mean):   {summary['ram_gb_mean']:.2f} GB")
            if 'gpu_power_w_mean' in summary:
                print(f"GPU Power (mean):   {summary['gpu_power_w_mean']:.1f} W")
                print(f"Total Energy:       {summary['cumulative_energy_wh']:.3f} Wh ({summary['cumulative_energy_kwh']:.6f} kWh)")
            if 'gpu_util_pct_mean' in summary:
                print(f"GPU Util (mean):    {summary['gpu_util_pct_mean']:.1f}%")
            print('='*60)
            
        if self._nvml_inited:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass