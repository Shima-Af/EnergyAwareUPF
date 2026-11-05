"""Corrected controller pilot, isolated from historical experiments.

SB3's TensorBoard integration can import TensorFlow. Initialize the frozen CPU
profile first, before any SB3 import, so inference cannot inherit another runtime.
"""
from revision.evaluation.twin import _runtime
_runtime()
