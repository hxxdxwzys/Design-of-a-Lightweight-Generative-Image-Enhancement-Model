# verify_shim.py
# ---------------------------------------------------------------------------
# Numerical verification of the standalone vae_shim package against the official
# diffusers 0.39 implementations, on the local GPU (fp32).
#
#   a) Official SD2.1 VAE (diffusers from_pretrained)   -> encode/decode -> out_ref
#   b) vae_shim.AutoencoderKL (same weights via load_state_dict) -> out_shim
#   c) MAE(out_ref, out_shim) must be < 1e-3
#   d) LightDecoder: official (diffusers Decoder rebuilt with sd21_lwdecoder.pth
#      weights) vs vae_shim.Decoder (same weights) -> MAE < 1e-3
#   e) Result (PASS/FAIL) written to D:/tmp_timing/verify_shim.log
#
# Small batches (2 x 256x256) per the environment note; two independent input
# batches are used so that 4 distinct images pass through both models in total.
# ---------------------------------------------------------------------------
import os
import sys
import time
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, r"D:/models/final_compliant/model_dir")

import torch  # noqa: E402

import vae_shim  # noqa: E402

LOG_PATH = r"D:/tmp_timing/verify_shim.log"
VAE_DIR = r"C:/models/finals/preset/ckpts/stable-diffusion-2-1-base/vae"
LD_PATH = r"C:/models/finals/preset/ckpts/sd21_lwdecoder.pth"
MAE_TOL = 1e-3

_log_lines = []


def log(msg):
    _log_lines.append(str(msg))
    print(msg, flush=True)


def mae(a, b):
    return (a - b).abs().mean().item()


def main():
    log("=" * 78)
    log("verify_shim: vae_shim vs official diffusers 0.39 (fp32)")
    log("=" * 78)
    log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu: {torch.cuda.get_device_name(0)}  "
            f"mem_alloc={torch.cuda.memory_allocated(0)/1e6:.0f}MB  "
            f"mem_reserved={torch.cuda.memory_reserved(0)/1e6:.0f}MB")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------- official VAE
    from diffusers.models import AutoencoderKL as OfficialAutoencoderKL

    log(f"loading official SD2.1 VAE from {VAE_DIR} ...")
    ref_vae = OfficialAutoencoderKL.from_pretrained(VAE_DIR).to(device).eval()
    log(f"official VAE params: {sum(p.numel() for p in ref_vae.parameters())/1e6:.2f}M")

    # ------------------------------------------------------------------ shim VAE
    shim_vae = vae_shim.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        latent_channels=4,
        norm_num_groups=32,
    )
    # key-structure assertion
    ref_keys = set(ref_vae.state_dict().keys())
    shim_keys = set(shim_vae.state_dict().keys())
    log(f"VAE state_dict keys: official={len(ref_keys)} shim={len(shim_keys)} "
        f"equal={ref_keys == shim_keys}")
    assert ref_keys == shim_keys, "state_dict key mismatch between official VAE and vae_shim"
    miss = shim_vae.load_state_dict(ref_vae.state_dict(), strict=True)
    log(f"VAE strict load_state_dict from official state dict: {miss}")
    shim_vae = shim_vae.to(device).eval()

    # --------------------------------------------------- official LightDecoder
    ld_ckpt = torch.load(LD_PATH, map_location="cpu", weights_only=True)
    cfg = ld_ckpt["config"]
    ld_state = ld_ckpt["model_state_dict"]
    log(f"LightDecoder config (from {os.path.basename(LD_PATH)}): {cfg}")

    from diffusers.models.autoencoders.vae import Decoder as OfficialDecoder

    ref_ld = OfficialDecoder(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg["layers_per_block"],
    ).to(device).eval()
    miss = ref_ld.load_state_dict(ld_state, strict=True)
    log(f"official LightDecoder strict load from sd21_lwdecoder.pth: {miss}")

    # ------------------------------------------------------------------- shim LD
    shim_ld = vae_shim.Decoder(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg["layers_per_block"],
    )
    ld_keys_ref = set(ref_ld.state_dict().keys())
    ld_keys_shim = set(shim_ld.state_dict().keys())
    log(f"LightDecoder state_dict keys: official={len(ld_keys_ref)} shim={len(ld_keys_shim)} "
        f"equal={ld_keys_ref == ld_keys_shim}")
    assert ld_keys_ref == ld_keys_shim, "state_dict key mismatch between official LightDecoder and vae_shim"
    miss = shim_ld.load_state_dict(ld_state, strict=True)
    log(f"shim LightDecoder strict load_state_dict: {miss}")
    shim_ld = shim_ld.to(device).eval()

    # ------------------------------------------------------------------- inputs
    def make_batch(seed, n=2, size=256):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return (torch.rand(n, 3, size, size, generator=g) * 2.0 - 1.0).to(device)

    batches = [make_batch(1001), make_batch(2002)]  # two independent 2x256 batches

    # -------------------------------------------------------------- comparisons
    all_maes = {}  # name -> float

    with torch.no_grad():
        for bi, x in enumerate(batches):
            log(f"\n--- input batch {bi}: shape={tuple(x.shape)} dtype={x.dtype} "
                f"range=[{x.min().item():.3f},{x.max().item():.3f}]")

            # encode: compare posterior statistics
            post_ref = ref_vae.encode(x).latent_dist
            post_shim = shim_vae.encode(x).latent_dist
            maes = {
                "encode.mean": mae(post_ref.mean, post_shim.mean),
                "encode.logvar": mae(post_ref.logvar, post_shim.logvar),
                "encode.std": mae(post_ref.std, post_shim.std),
            }
            for k, v in maes.items():
                all_maes[f"b{bi}.{k}"] = v
                log(f"  MAE {k:16s} = {v:.3e}")

            # decode (mode, deterministic)
            z_mode = post_ref.mode()
            out_ref = ref_vae.decode(z_mode).sample
            out_shim = shim_vae.decode(z_mode).sample
            v = mae(out_ref, out_shim)
            all_maes[f"b{bi}.decode_mode"] = v
            log(f"  MAE decode(mode)      = {v:.3e}   (ref range [{out_ref.min().item():.3f},{out_ref.max().item():.3f}])")

            # decode (random posterior sample, same RNG seed on both sides)
            g_ref = torch.Generator(device=device).manual_seed(777)
            z_ref = post_ref.sample(g_ref)
            g_shim = torch.Generator(device=device).manual_seed(777)
            z_shim = post_shim.sample(g_shim)
            out_ref_s = ref_vae.decode(z_ref).sample
            out_shim_s = shim_vae.decode(z_shim).sample
            v = mae(out_ref_s, out_shim_s)
            all_maes[f"b{bi}.decode_sample"] = v
            log(f"  MAE decode(sample)    = {v:.3e}")

            # end-to-end forward()
            f_ref = ref_vae(x).sample
            f_shim = shim_vae(x).sample
            v = mae(f_ref, f_shim)
            all_maes[f"b{bi}.forward"] = v
            log(f"  MAE forward()         = {v:.3e}")

            # ---------------------------------------------------- LightDecoder
            # input 1: real encoder latent (mode)
            z_ld = z_mode
            ld_ref = ref_ld(z_ld)
            ld_shim = shim_ld(z_ld)
            v = mae(ld_ref, ld_shim)
            all_maes[f"b{bi}.ld_encode_latent"] = v
            log(f"  MAE LightDecoder(real latent) = {v:.3e}")

            # input 2: random normal latent (seeded, same on both sides)
            gz = torch.Generator(device=device).manual_seed(42 + bi)
            z_rand = torch.randn(2, cfg["in_channels"], 32, 32, generator=gz, device=device)
            ld_ref2 = ref_ld(z_rand)
            ld_shim2 = shim_ld(z_rand)
            v = mae(ld_ref2, ld_shim2)
            all_maes[f"b{bi}.ld_random_latent"] = v
            log(f"  MAE LightDecoder(random latent)= {v:.3e}")

            del post_ref, post_shim, out_ref, out_shim
            if device == "cuda":
                torch.cuda.empty_cache()

    worst = max(all_maes.values())
    worst_name = max(all_maes, key=all_maes.get)
    log("\n" + "-" * 78)
    log(f"worst MAE over all comparisons: {worst_name} = {worst:.3e}  (tolerance {MAE_TOL})")
    passed = worst < MAE_TOL
    log("-" * 78)
    log("RESULT: " + ("PASS" if passed else "FAIL"))
    log(f"(fp32, batches=2x2x256x256, tolerance={MAE_TOL})")
    return passed


if __name__ == "__main__":
    t0 = time.time()
    ok = False
    try:
        ok = bool(main())
    except Exception:
        log("\nEXCEPTION during verification:\n" + traceback.format_exc())
        ok = False
    finally:
        log(f"\nfinished in {time.time() - t0:.1f}s")
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(_log_lines) + "\n")
        print(f"\nlog written to {LOG_PATH}")
    sys.exit(0 if ok else 1)
