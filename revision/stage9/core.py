"""Training-only trace transformations and complete scalar-ReLU geometry."""
import numpy as np


def segments(loads, times, name, spec):
    """No final suffix accepted; selection never inspects validation values."""
    if len(loads) != 2520 or len(times) != 2520:
        raise ValueError('Only the authorized 2520-row prefix is accepted')
    train = np.arange(96, 2016); val = np.arange(2016, 2520)
    kind = spec['kind']
    if kind == 'rapid':
        width = spec['window_slots']
        # Maximize within-window variation, not artificial boundary jumps.
        candidates = range(96, 2016 - width + 1)
        start = max(candidates, key=lambda i: (float(np.abs(np.diff(loads[i:i+width])).sum()), -i))
        clock = (times[start].hour, times[start].minute)
        vstart = next(i for i in range(2016, 2520-width+1)
                      if (times[i].hour, times[i].minute) == clock)
        train = np.arange(start, start+width); val = np.arange(vstart, vstart+width)
    result = {}
    for role, indices in [('train', train), ('validation', val)]:
        if kind == 'mixed':
            factors = np.array([spec['factors'][(int(i)//spec['block_slots']) % len(spec['factors'])] for i in indices])
        else:
            factors = np.full(len(indices), spec['factor'])
        original = np.asarray(loads)[indices]
        result[role] = dict(indices=indices, factors=factors, loads=original*factors,
                            original=original, timestamps=[times[i].isoformat() for i in indices],
                            joins=np.r_[False, factors[1:] != factors[:-1]])
    return result


def relu_pieces(network, scaler, domain):
    """Propagate exact-real (float64 coefficient) affine pieces through the frozen NN.

    All hidden nonlinearities are ReLU; BatchNorm is affine at inference and
    dropout is identity. Return physical-load pieces of the final logit. This
    identifies all threshold crossings structurally rather than trusting a grid.
    Floating inference around crossings is checked separately in the driver.
    """
    lo, hi = (np.log1p(np.asarray(domain))-scaler['mean'])/scaler['scale']
    pieces = [(float(lo), float(hi), np.ones(1), np.zeros(1))]
    for layer in network.layers:
        kind = type(layer).__name__
        if kind in ('InputLayer', 'Dropout'):continue
        if kind == 'BatchNormalization':
            gamma, beta, mean, variance = [np.asarray(w, float) for w in layer.get_weights()]
            scale = gamma / np.sqrt(variance+layer.epsilon)
            pieces = [(l, h, m*scale, (b-mean)*scale+beta) for l,h,m,b in pieces]
            continue
        if kind != 'Dense':raise ValueError(f'Unsupported scalar geometry: {kind}')
        weight, bias = [np.asarray(w, float) for w in layer.get_weights()]
        affine = [(l, h, m@weight, b@weight+bias) for l,h,m,b in pieces]
        activation = layer.get_config()['activation']
        if activation in ('linear', 'sigmoid'):
            pieces = affine
        elif activation == 'relu':
            pieces = []
            for l,h,m,b in affine:
                roots = [-bv/mv for mv,bv in zip(m,b) if mv != 0 and l < -bv/mv < h]
                edges = sorted(set([l,h]+roots))
                for a,z in zip(edges,edges[1:]):
                    mask = (m*((a+z)/2)+b) > 0
                    pieces.append((a,z,m*mask,b*mask))
        else:raise ValueError('Unsupported activation')
    if any(len(m) != 1 for _,_,m,_ in pieces):raise ValueError('Expected scalar logit')
    return pieces


def qos_roots(model, threshold=.90):
    scaler = model.metadata['input_scaler']; domain = model.metadata['domain_mbps']
    pieces = relu_pieces(model.network, scaler, domain)
    target = np.log(threshold/(1-threshold))
    roots = []
    for l,h,m,b in pieces:
        if m[0] != 0:
            z = (target-b[0])/m[0]
            if l <= z <= h:
                load = float(np.expm1(z*scaler['scale']+scaler['mean']))
                if not roots or abs(load-roots[-1]) > 1e-8:
                    roots.append(load)
    return roots, pieces


def distribution(x):
    x = np.asarray(x, float)
    return dict(min=float(x.min()), max=float(x.max()), mean=float(x.mean()),
                **{f'p{p:02d}':float(np.percentile(x,p)) for p in [1,5,25,50,75,95,99]})
