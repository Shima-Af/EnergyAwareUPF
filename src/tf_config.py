"""
TensorFlow configuration module to suppress warnings and set environment variables.
Import this module at the beginning of any script that uses TensorFlow.
"""
import os

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO+WARNING, 3=no INFO+WARNING+ERROR

# Additional TensorFlow optimizations (optional)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU only if you want to force CPU usage

print("\033[1;91m✓ TensorFlow configuration applied: warnings suppressed\033[0m")
