"""Evaluate/plot isolated campaign using the frozen analysis and environment."""
import argparse
from revision.evaluation.twin import ROOT
from . import evaluate,analysis
from .evidence import write_comparison_csv


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['evaluate','plot']);args=parser.parse_args()
    module=evaluate if args.operation=='evaluate' else analysis
    module.OUT=ROOT/'revision/reports/stage4_rng_isolated';module.MODELS=ROOT/'revision/models/stage4_rng_isolated'
    if args.operation=='plot':module.write_csv=write_comparison_csv
    module.main()

if __name__=='__main__':main()
