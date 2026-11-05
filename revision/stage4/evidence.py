"""Render the frozen analysis with an explicit union schema for mixed controller rows.

Static rows have no selected checkpoint; learned rows do. The shared historical
CSV helper takes only the first row's keys, so provide a complete schema before
calling it. No numeric value, model selection, or frozen source is changed.
"""
from revision.evaluation.validate_static import write_csv
from . import analysis


def write_comparison_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    write_csv(path,[{k:row.get(k) for k in keys} for row in rows])


def main():
    analysis.write_csv=write_comparison_csv
    analysis.main()

if __name__=='__main__':main()
