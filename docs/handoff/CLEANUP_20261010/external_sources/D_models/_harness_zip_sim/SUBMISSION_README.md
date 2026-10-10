# Track2 Submission — LiteDiff-Enhance

Contents (per the review notice):
  1) Inference project : `model_dir/`, `infer.py`, `selfcheck.py`
  2) Model weights     : `model_dir/your_model.pt` (TorchScript, fp16)
  3) Dependencies      : `requirements.txt`
  4) One-page proposal : `Proposal_OnePage.pptx`
  (extra) `samples/`   : 2 input/output pairs from this exact package, for the byte-exact self-check
  (extra) `README_CN.md`: Chinese description (architecture / env / run / compliance)

Quick check (needs torch + a CUDA GPU):
  python selfcheck.py
    -> checks the 512x512 harness interface and reproduces the bundled 4K samples

Harness interface: `Runner(model_dir).infer(image_tensor, prompt="")`, in/out `[1,3,H,W]` in `[-1,1]`.
The 4K images are produced by the SAME `model_dir` weights and the SAME code path,
with only the patch/overlap configuration differing (explicitly allowed by the rules).
