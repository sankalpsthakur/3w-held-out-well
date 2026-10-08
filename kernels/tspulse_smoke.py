# Kernel entry: can we install tsfm_public and run TSPulse reconstruction on this image? Introspection only, synthetic input, no 3W data.
import inspect, json, subprocess, sys, time
out = {"t0": time.time()}
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "granite-tsfm"], capture_output=True, text=True)
out["pip_rc"] = r.returncode
out["pip_tail"] = (r.stdout + r.stderr)[-1500:]
try:
    import torch, transformers
    out["torch"] = torch.__version__; out["transformers"] = transformers.__version__
    import tsfm_public
    out["tsfm_public"] = getattr(tsfm_public, "__version__", "?")
    from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
    out["forward_signature"] = str(inspect.signature(TSPulseForReconstruction.forward))
    C = 1
    model = TSPulseForReconstruction.from_pretrained("ibm-granite/granite-timeseries-tspulse-r1", num_input_channels=C, revision="main", mask_type="user")
    model.eval()
    out["n_params"] = sum(p.numel() for p in model.parameters())
    cfg = model.config
    out["config"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)[:80]) for k, v in cfg.to_dict().items() if k in (
        "context_length", "patch_length", "num_input_channels", "d_model", "num_layers", "mask_type", "scaling", "mode", "decoder_mode", "prediction_length")}
    x = torch.randn(4, 512, C)
    t = time.time()
    with torch.no_grad():
        o = model(past_values=x)
    out["forward_seconds_batch4"] = time.time() - t
    out["output_type"] = type(o).__name__
    out["output_fields"] = {k: list(v.shape) for k, v in (o.items() if hasattr(o, "items") else []) if hasattr(v, "shape")}
    xb = torch.randn(256, 512, C)
    torch.set_num_threads(4)
    t = time.time()
    with torch.no_grad():
        model(past_values=xb)
    out["forward_seconds_batch256_univariate_4threads"] = time.time() - t
    # forecast-style mask: last 16 steps unobserved
    mask = torch.ones(256, 512, C); mask[:, -16:, :] = 0
    t = time.time()
    with torch.no_grad():
        o2 = model(past_values=xb, past_observed_mask=mask)
    out["forward_seconds_masked_batch256"] = time.time() - t
    out["masked_output_fields"] = {k: list(v.shape) for k, v in (o2.items() if hasattr(o2, "items") else []) if hasattr(v, "shape")}
    rec = o2.get("reconstruction_outputs") if hasattr(o2, "get") else None
    if rec is not None:
        out["recon_shape"] = list(rec.shape)
        out["masked_region_mse_random_input"] = float(((rec[:, -16:, :] - xb[:, -16:, :]) ** 2).mean())
        out["observed_region_mse_random_input"] = float(((rec[:, :-16, :] - xb[:, :-16, :]) ** 2).mean())
except Exception as e:
    import traceback
    out["error"] = repr(e)[:500]; out["tb"] = traceback.format_exc()[-1500:]
json.dump(out, open("/kaggle/working/tspulse_smoke.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
