"""Exact stateful bounds using the same transition-aware engine as controllers."""
from fractions import Fraction
from revision.evaluation.contract import State


def solve(engine,loads,objective,*,actions=(0,1,2),gamma=.995):
    if objective not in ('reward','energy','qos_constrained_energy'):raise ValueError('Unknown oracle')
    if tuple(sorted(set(actions)))!=actions or 0 not in actions or any(a not in (0,1,2) for a in actions):raise ValueError('Invalid action set')
    if not loads or not 0<gamma<=1:raise ValueError('Invalid horizon/discount')
    states=[State(a,c) for a in actions for c in range(5)]
    costs={};denominator=1
    for t,load in enumerate(loads):
        for s in states:
            for a in actions:
                if a!=s.action and s.elapsed_slots<4:continue
                step=engine.step(s,a,load)
                if objective=='qos_constrained_energy' and not step['compliant']:continue
                v=-gamma**t*step['reward'] if objective=='reward' else step['total_wh']
                n,d=v.as_integer_ratio();denominator=max(denominator,d)
                costs[t,s,a]=(n,d,step['transition'].next_state)
    future={s:(0,0) for s in states};policy={}
    for t in reversed(range(len(loads))):
        current={}
        for s in states:
            candidates=[]
            for a in actions:
                c=costs.get((t,s,a))
                if c is None or c[2] not in future:continue
                n,d,next_state=c;v,changes=future[next_state]
                candidates.append((n*(denominator//d)+v,changes+(a!=s.action),a,next_state))
            if candidates:
                best=min(candidates,key=lambda c:c[:3]);current[s]=best[:2];policy[t,s]=best[2:]
        future=current
    state=State(0,4)
    if state not in future:return dict(feasible=False,actions=[],objective_value=None)
    value,changes=future[state];seq=[]
    for t in range(len(loads)):
        a,state=policy[t,state];seq.append(a)
    exact=Fraction(-value if objective=='reward' else value,denominator)
    return dict(feasible=True,actions=seq,objective_value=float(exact),reconfigurations=changes,
                exact_numerator=exact.numerator,exact_denominator=exact.denominator)
