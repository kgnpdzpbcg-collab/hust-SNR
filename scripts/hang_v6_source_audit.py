"""Source-only, label-collision audit for Hang trajectory representations.

Do not inspect Target-Dev while choosing feature preprocessing. This checks
whether two fault labels map to exact identical 40x40 input images, which no
deterministic classifier can distinguish.
"""
from collections import defaultdict, Counter
import hashlib
import numpy as np


def image_collision_audit(inputs, labels):
    inputs=np.asarray(inputs)
    labels=np.asarray(labels)
    if len(inputs)!=len(labels):
        raise ValueError("length mismatch")
    hashes=[hashlib.blake2b(np.asarray(x).tobytes(),digest_size=16).digest()
            for x in inputs]
    groups=defaultdict(set)
    for h,y in zip(hashes,labels):
        groups[h].add(int(y))
    ambiguous={h:tuple(sorted(s)) for h,s in groups.items() if len(s)>1}
    affected=sum(h in ambiguous for h in hashes)
    return dict(samples=int(len(labels)), unique_images=len(groups),
                cross_label_identical_images=len(ambiguous),
                affected_samples=int(affected),
                affected_fraction=float(affected/max(len(labels),1)),
                ambiguous_label_groups={
                    ",".join(map(str,k)):int(v) for k,v in sorted(
                        Counter(ambiguous.values()).items()
                    )
                })


def compare_hang_normalizers(meta,raw,labels,config):
    from compare_jung_hang_v5 import build_inputs
    out={}
    for mode in ("pointwise_eq6","reference_radius_figures"):
        cfg={**config,"hang_normalization_mode":mode}
        images,indices=build_inputs("hang",meta,raw,cfg)
        out[mode]=image_collision_audit(images,np.asarray(labels)[indices])
    return out
