"""One energy-based reward normalization, using only original training powers.

Does not substitute a loss value or evaluate controllers. No SEC denominator.
"""
import csv,json,math
from dataclasses import asdict
from revision.evaluation.twin import ROOT,sha256
from revision.evaluation.contract import State,actuate,Contract
from revision.stage10.primitives import activation_burst_wh,transition_energy_wh
from .engine import Dynamics
from .independent_fits import OUT,CFG,dump


def main():
    source=ROOT/'revision/reports/stage9/scenario_samples.csv'
    names={'original':'original','low_peak40_reference':'low',
           'high_peak800_reference':'moderate_overload','mixed_daily':'mixed'}
    d=Dynamics(.82,.015,0.);energies=[];per_scenario={}
    with source.open() as f:
        for row in csv.DictReader(f):
            if row['workload']!='netflix' or row['role']!='train' or row['scenario'] not in names:continue
            values=[];p=[float(row[f'power_{a}_w']) for a in range(3)]
            for old in range(3):
                for new in range(3):
                    tr=actuate(State(old,4),new,Contract());duration=0.;burst=0.
                    if old!=new:
                        if new==0:
                            duration=24.;burst=activation_burst_wh(burst_ratio=1.73,attributed_idle_w=.82,per_instance_seconds=24.,launched_instances=1)
                        elif tr.usr_instances_launched:
                            duration=tr.usr_instances_launched*3.3
                            burst=activation_burst_wh(burst_ratio=.37,attributed_idle_w=.015,per_instance_seconds=3.3,launched_instances=tr.usr_instances_launched)
                    e=transition_energy_wh(outgoing_w=p[old],incoming_w=p[new],outgoing_seconds=duration,slot_seconds=900.,activation_wh=burst)['total_wh']
                    e+=d.usr_idle_w*d.usr_seconds_per_instance*tr.usr_instances_launched*(tr.usr_instances_launched-1)/2/3600
                    values.append(e)
            name=names[row['scenario']];per_scenario.setdefault(name,[]).extend(values);energies.extend(values)
    assert len(energies)==4*1920*9
    scale=max(energies)
    dump(OUT/'reward_calibration_v2.json',dict(
       accounting_amendment='Serial-launch first-ready replica idle overlap included; no controller or model fit changed; initial reward_calibration.json preserved',
       status='Single prepared reward calibrated without any controller outcome; full four-KPI execution remains gated',
       energy_scale_wh=scale,role='train',training_identity=sha256(source),
       selected_scenarios=list(names.values()),source='Stage9 frozen power columns ONLY: Netflix training rows96..2015',
       consulted_source_fields=['workload','role','scenario','power_0_w','power_1_w','power_2_w'],
       calibration_transitions=len(energies),dynamics=asdict(d),protocol=CFG['reward_formula_prepared'],
       coefficients={'energy':1.,'strict':3.,'bounded_severity':1.,'stability':.01},
       strict_weight_derivation='ceil(energy_bound(1)+bounded_severity_bound(1)+stability_bound(.01))=3; minimum strict-unsafe cost3 exceeds maximum safe immediate cost1.01 inside training energy bound',
       no_sec=True,activation_energy_included_once=True,old_launch_reward_penalties_used=False,
       stability_units='dimensionless preference cost per executed reconfiguration, 1% of maximum training slot energy cost; not measured Wh',
       full_reward_diagnostics_complete=False,unknown_component='Physical loss-ratio predictions',
       training_energy={name:{'n':len(x),'max_wh':max(x),'mean_wh':math.fsum(x)/len(x)} for name,x in per_scenario.items()},
       operational_guarantee=False,scope='Point-model immediate priority within calibrated training bound, not whole-horizon feasibility or unseen-load assurance'))


if __name__=='__main__':main()
