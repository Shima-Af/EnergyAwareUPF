"""Run the unchanged full-budget trainer with RNG-isolated checkpoint validation.

Separate output roots retain every initial affected seed and all checkpoints.
Only the checkpoint capture wrapper changes; all model/algorithm/settings and
forecast bytes remain identical. Shared frozen source files are not edited.
"""
from revision.evaluation.twin import ROOT
from . import train
from .rng_isolation import preserve_rng


def main():
    train.OUT=ROOT/'revision/reports/stage4_rng_isolated'
    train.MODELS=ROOT/'revision/models/stage4_rng_isolated'
    train.ConvergenceCallback.capture=preserve_rng(train.ConvergenceCallback.capture)
    train.main()

if __name__=='__main__':main()
