"""V3 experiment CLI. Reuse exact frozen V2 source/noise/Dev pipelines.

Training and checkpoint selection are source-only. Decoder MSE is applied
only during training and never changes the number of inputs at inference.
"""
import csv
import json
import sys
from pathlib import Path

import v2_experiment as core
from v3_cae_prior import (METHODS,CAEFrequencyClassifier,paired_cae_objective,
                          count_parameters,smoke as model_smoke)

# V2 is the validated end-to-end runner. Replace ONLY architecture and loss,
# not record splits, Gaussian RNG, raw interpolation or evaluation protocol.
core.METHODS=METHODS
core.HarmonicCNN=CAEFrequencyClassifier
core.counts_by_module=lambda m: {"cnn":count_parameters(m)["encoder_classifier"],
    "decoder":count_parameters(m)["decoder"],
    "spectral":count_parameters(m)["spectral"],"total":count_parameters(m)["total"]}
core.paired_cross_entropy=lambda model,clean,noisy,label: paired_cae_objective(
    model,clean,noisy,label,lambda_rec=1.0)[0]
core.DEFAULT_OUT=Path("outputs/v3_cae_prior")

original_training_run=core.training_run
original_smoke=core.smoke


def v3_training_run(mode,seed,outdir,epochs=50,batch_size=32,lr=.001):
    original_training_run(mode,seed,outdir,epochs,batch_size,lr)
    p=Path(outdir)/"models"/mode/f"seed{seed}"/"run.json"
    d=json.loads(p.read_text())
    d.update({"experiment":"V3_CAE_LOW_PASS_PRIOR","lambda_rec":1.0,
        "decoder_training_only": mode!="cae_prior_no_rec",
        "source_only":True,"final_holdout_evaluated":False})
    p.write_text(json.dumps(d,indent=2))


def v3_smoke():
    model_smoke()
    original_smoke()
    print("V3 end-to-end smoke PASSED")


core.training_run=v3_training_run
core.smoke=v3_smoke


def main():
    command=sys.argv[1] if len(sys.argv)>1 else ""
    core.main()
    if command!="summarize":
        return
    # The unchanged V2 summary computes all per-condition means and stds.
    # Add source-independent paired contrasts for the V3 research question.
    path=Path("outputs/v3_cae_prior")
    if "--output" in sys.argv:
        path=Path(sys.argv[sys.argv.index("--output")+1])
    p=path/"SUMMARY.json"
    report=json.loads(p.read_text())
    report["status"]="V3_dev_only"
    with (path/"PER_SEED_RESULTS.csv").open(newline="") as fd:
        rows=list(csv.DictReader(fd))
    report["paired_contrasts"]=[]
    for condition in sorted(set(r["condition"] for r in rows)):
        for comparator in ("cae_identity","cae_lowpass3","cae_prior_no_rec"):
            differences=[]
            for seed in sorted(set(r["seed"] for r in rows)):
                def find(mode):
                    return next((float(r["macro_f1"]) for r in rows
                        if r["condition"]==condition and r["mode"]==mode
                        and r["seed"]==seed),None)
                proposed,baseline=find("cae_prior"),find(comparator)
                if proposed is not None and baseline is not None:
                    differences.append(proposed-baseline)
            if differences:
                report["paired_contrasts"].append({"condition":condition,
                    "prior_minus":comparator,"delta_pp_by_seed":differences,
                    "delta_mean_pp":sum(differences)/len(differences)})
    p.write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
