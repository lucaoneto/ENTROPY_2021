
# FAIRNESS METRICS - Binary Classification for Binary Sensitive Distributions
def DDP(d1, d2):
    d1_uni, d2_uni = d1.unique(), d2.unique()
    outputs = d1_uni if len(d1_uni)>len(d2_uni) else d2_uni

    assert len(outputs) in (1,2), "Non binary outputs"

    if len(outputs)==1: return 100.*(d1_uni.item()!=d2_uni.item())
    # if len(outputs)==1: return 100.-100.*(d2==outputs.item()).sum()/len(d2)

    # assert all(outputs==d2.unique()), "the two distributions don't have same binary values"
    assert all([e in outputs for e in d1_uni]) and all([e in outputs for e in d2_uni]), "the two distributions don't have same binary values"

    d1_posRate = 100.*(d1==outputs[1]).sum()/len(d1)
    d2_posRate = 100.*(d2==outputs[1]).sum()/len(d2)
    return ((d1_posRate-d2_posRate).abs()).item()


# FAIRNESS METRICS - Regression for Binary Sensitive Distributions
from torch import cat, stack

def sum_cdfDDP(d1, d2):
    sort_conc = cat((d1,d2), dim=0).sort().values
    edf_d1 = stack([(d1<=v).sum(dtype=float)/len(d1) for v in sort_conc])
    edf_d2 = stack([(d2<=v).sum(dtype=float)/len(d2) for v in sort_conc])
    return ((edf_d1-edf_d2).abs().sum()).item()

def max_cdfDDP(d1, d2):
    sort_conc = cat((d1,d2), dim=0).sort().values
    edf_d1 = stack([(d1<=v).sum(dtype=float)/len(d1) for v in sort_conc])
    edf_d2 = stack([(d2<=v).sum(dtype=float)/len(d2) for v in sort_conc])
    return ((edf_d1-edf_d2).abs().max()).item()