# Kernel entry: zero-shot TSPulse event detector on 3W-HW v1 (PREREGISTRATION-3W-TSPULSE.md). Event detection only (no class output), no training.
#   per scored decision and per process channel: the last 3,584 rows are block-averaged into 512 points (7 rows each), z-normalised with the first 504 points only, the last patch (8 points, 56 s) is masked, TSPulse
#   reconstructs it, the error on that patch is the channel score; channels are scaled by the dev-normal 99th percentile, combined with a max, mapped to [0, 1) with v / (v + 1); scores are held (forward-filled) between scored decisions.
#   phase 1 (dev wells only): errors, per-channel references, dev alternatives (information only), threshold; phase 2: errors of the 10 sealed wells and ONE scoring.
# cfg["smoke"]: code-path and timing check on dev wells only (dev fold 3 plays the test); no sealed-test well is loaded.
import hashlib, json, os, subprocess, sys, time, warnings
import numpy as np
from w3 import features as F, pipeline as P, scorer as S

OUT = "/kaggle/working"
cfg = json.load(open("/tmp/w3code/data/config.json"))
SMOKE = bool(cfg.get("smoke"))
T0 = time.time()
warnings.filterwarnings("ignore")
results = {"config": cfg, "smoke": SMOKE}


def dump():
    json.dump(P.jsonable(results), open(f"{OUT}/results.json", "w"))


# ---- install and load the model (public checkpoint, downloaded inside the kernel)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "granite-tsfm"], capture_output=True, text=True)
results["pip_rc"] = r.returncode
if r.returncode != 0:
    results["pip_tail"] = (r.stdout + r.stderr)[-1500:]
    dump()
    raise SystemExit("granite-tsfm could not be installed: " + (r.stdout + r.stderr)[-500:])
import torch, transformers
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
torch.set_num_threads(4)
REPO = "ibm-granite/granite-timeseries-tspulse-r1"
try:
    from huggingface_hub import HfApi
    results["model_sha"] = HfApi().model_info(REPO).sha
except Exception as e:  # noqa: BLE001
    results["model_sha"] = "unknown: " + repr(e)[:200]
try:
    model = TSPulseForReconstruction.from_pretrained(REPO, num_input_channels=1, revision=cfg["model_revision"], mask_type="user").eval()
    results["model_revision_loaded"] = cfg["model_revision"]
except Exception as e:  # noqa: BLE001
    results["model_revision_error"] = repr(e)[:300]
    model = TSPulseForReconstruction.from_pretrained(REPO, num_input_channels=1, revision="main", mask_type="user").eval()
    results["model_revision_loaded"] = "main (fallback), sha " + str(results.get("model_sha"))
results["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
                       "n_params": int(sum(p.numel() for p in model.parameters())), "context_length": int(model.config.context_length), "patch_length": int(model.config.patch_length)}
L, D, MASK_PTS = 512, 7, 8
assert model.config.context_length == L and model.config.patch_length == MASK_PTS, results["versions"]
W_ROWS = L * D                       # 3,584 rows of lookback (the protocol allows 3,600)
STRIDE_DEC = int(cfg["stride_dec"])  # a window is scored every STRIDE_DEC decisions (30 s each)
BATCH = 256
COUNT = {"calls": 0, "model_seconds": 0.0, "windows": 0}
P.log(f"model loaded: {results['versions']}")

# ---- data
bench = S.Benchmark.load("/tmp/w3code/data")
index = json.load(open("/tmp/w3code/data/index_labels.json"))
files = sorted(index["files"], key=lambda f: f["path"])
assert [f["path"] for f in files] == bench.inst_path
assert hashlib.sha256(open("/tmp/w3code/data/manifest.csv", "rb").read()).hexdigest() == cfg["manifest_sha256"], "manifest differs from the revealed one"
for k, ref in cfg["decision_array_sha256"].items():
    assert P.array_digest(getattr(bench, k)) == ref, f"decision table array {k} differs"
dev, test = bench.wells_with_role("dev"), bench.wells_with_role("test")
if SMOKE:
    d1, d2, d3 = (bench.wells_with_role("dev", k) for k in (1, 2, 3))
    dev, test = d1 + d2, d3
    if cfg.get("smoke_wells"):
        dev, test = dev[:cfg["smoke_wells"]], test[:max(1, cfg["smoke_wells"] // 2)]
assert not (set(dev) & set(test))
if SMOKE:
    assert not (set(dev) | set(test)) & set(bench.wells_with_role("test")), "a sealed-test well is in the smoke set"
classes_eval = P.dev_classes(bench)
B = 50 if SMOKE else S.BOOT_B
NCH = len(F.CONT)


def block_windows(x, rows):
    """x [n_rows] float64 (NaN = missing); rows = decision rows. Returns [len(rows), 512] block means of rows r-3583..r (NaN where a whole block is missing)."""
    idx = rows[:, None] - (W_ROWS - 1) + np.arange(W_ROWS)[None, :]
    return np.nanmean(x[idx].reshape(len(rows), L, D), axis=2)


def channel_errors(cont, rows):
    """[len(rows), 18] masked-patch reconstruction error in z units; NaN where the channel has too many missing points or is absent."""
    err = np.full((len(rows), NCH), np.nan, dtype=np.float32)
    for c in range(NCH):
        x = cont[:, c].astype(np.float64)
        if not np.isfinite(x).any():
            continue
        bw = block_windows(x, rows)
        fin = np.isfinite(bw)
        ok = fin.mean(1) >= 0.9
        if not ok.any():
            continue
        bw = bw[ok]; fin = fin[ok]
        past = np.where(fin[:, :L - MASK_PTS], bw[:, :L - MASK_PTS], np.nan)
        mu = np.nanmean(past, axis=1, keepdims=True)
        sd = np.nanstd(past, axis=1, keepdims=True)
        sd = np.maximum(sd, 1e-6 * (np.abs(mu) + 1.0))
        z = np.where(fin, (bw - mu) / sd, 0.0)                        # a missing point is the window mean (z = 0)
        z = np.clip(z, -20.0, 20.0).astype(np.float32)
        out = np.zeros(len(z), dtype=np.float32)
        for a in range(0, len(z), BATCH):
            xb = torch.from_numpy(z[a:a + BATCH, :, None].copy())
            m = torch.ones_like(xb); m[:, -MASK_PTS:, :] = 0
            t_m = time.time()
            with torch.no_grad():
                o = model(past_values=xb, past_observed_mask=m)
            COUNT["model_seconds"] += time.time() - t_m; COUNT["calls"] += len(xb)
            rec = o["reconstruction_outputs"] if hasattr(o, "keys") else o.reconstruction_outputs
            out[a:a + BATCH] = ((rec[:, -MASK_PTS:, 0] - xb[:, -MASK_PTS:, 0]) ** 2).mean(1).numpy()
        err[np.flatnonzero(ok), c] = out
    return err


def scored_errors(wells, tag):
    """Errors at every STRIDE_DEC-th decision of every instance of `wells`. Returns dict inst -> (ks, err)."""
    inst = [i for i in range(len(files)) if bench.wells[bench.inst_well[i]] in set(wells)]
    out, t = {}, time.time()
    from concurrent.futures import ThreadPoolExecutor
    n_win = 0
    with ThreadPoolExecutor(4) as ex:
        for a in range(0, len(inst), 16):
            chunk = inst[a:a + 16]
            for i, (cont, _state) in zip(chunk, ex.map(lambda j: P.load_signals(files[j]), chunk)):
                off, end = bench.inst_offset[i], bench.inst_offset[i + 1]
                ks = np.arange(0, end - off, STRIDE_DEC)
                out[i] = (ks, channel_errors(cont, bench.r[off + ks].astype(np.int64)))
                n_win += len(ks); COUNT["windows"] += len(ks)
            P.log(f"  {tag}: {a + len(chunk)}/{len(inst)} instances, {n_win} scored decisions, {time.time() - t:.0f}s")
    return out


def aggregate(e, mode):
    """e [n, 18] channel scores (NaN = channel not valid at that window) -> [n] combined value (0 where no channel is valid)."""
    fin = np.isfinite(e)
    if mode in ("max_q", "max_raw"):
        v = np.where(fin, e, -np.inf).max(axis=1)
    elif mode == "mean_all_q":
        cnt = fin.sum(1)
        v = np.where(cnt > 0, np.where(fin, e, 0.0).sum(1) / np.maximum(cnt, 1), -np.inf)
    elif mode == "mean_top3_q":
        top = -np.sort(-np.where(fin, e, -np.inf), axis=1)[:, :3]
        f3 = np.isfinite(top)
        v = np.where(f3.any(1), np.where(f3, top, 0.0).sum(1) / np.maximum(f3.sum(1), 1), -np.inf)
    else:
        raise ValueError(mode)
    return np.where(np.isfinite(v), np.maximum(v, 0.0), 0.0)


def to_scores(err_by_inst, q, mode="max_q"):
    """Event score over ALL decisions (NaN outside the instances present): v / (v + 1), held between scored decisions (decision k uses the latest scored decision <= k)."""
    score = np.full(bench.n, np.nan, dtype=np.float64)
    for i, (ks, err) in err_by_inst.items():
        off, end = bench.inst_offset[i], bench.inst_offset[i + 1]
        v = aggregate(err if mode == "max_raw" else err / q[None, :], mode)
        score[off:end] = (v / (v + 1.0))[np.arange(end - off) // STRIDE_DEC]
    return score


# ---------------------------------------------------------------- phase 1: dev wells only
dev_err = scored_errors(dev, "dev")
results["seconds_dev_errors"] = time.time() - T0
results["dev_error_counts"] = dict(COUNT)
normal_err = []
for i, (ks, err) in dev_err.items():
    off = bench.inst_offset[i]
    m = bench.elig[off + ks] & (bench.y[off + ks] == 0)
    normal_err.append(err[m])
normal_err = np.concatenate(normal_err)
q = np.nanpercentile(normal_err, 99, axis=0)
q = np.where(np.isfinite(q) & (q > 0), q, np.nan)
results["q99_dev_normal"] = {c: (None if not np.isfinite(v) else float(v)) for c, v in zip(F.CONT, q)}
results["n_dev_normal_windows"] = int(len(normal_err))
P.log(f"dev errors done {time.time() - T0:.0f}s; q99 per channel: {np.round(q, 2).tolist()}")
dev_alt = {}
for mode in ("max_q", "mean_top3_q", "mean_all_q", "max_raw"):                 # information only; the primary is max_q, fixed in the preregistration
    sc = to_scores(dev_err, q, mode)
    pt = S.report(bench, sc, None, dev, None, boot=False)["point"]
    dev_alt[mode] = {"auroc": pt["auroc"], "average_precision": pt["average_precision"]}
    if mode == "max_q":
        score_dev = sc
results["dev_variants_information_only"] = dev_alt
theta, table = S.select_threshold(bench, score_dev, dev)
results["theta"] = theta
results["dev_primary"] = S.report(bench, score_dev, None, dev, theta, B=B)
dump()
P.log(f"phase 1 done {time.time() - T0:.0f}s: dev AUROC {dev_alt['max_q']['auroc']:.4f}, AP {dev_alt['max_q']['average_precision']:.4f}, theta {theta}; variants {dev_alt}")

# ---------------------------------------------------------------- phase 2: the test wells, one scoring
test_err = scored_errors(test, "test")
P.log(f"=== SEALED-TEST SCORING tspulse starts {time.strftime('%H:%M:%S')} ===")
all_err = {**dev_err, **test_err}
score = to_scores(all_err, q, "max_q")
rep = {"dev_oof": results["dev_primary"], "test": S.report(bench, score, None, test, theta, B=B)}
results["reports"] = rep
np.savez_compressed(f"{OUT}/preds_tspulse.npz", score=np.where(np.isfinite(score), score, 0.0).astype(np.float32))
results["seconds"] = time.time() - T0
results["all_error_counts"] = dict(COUNT)
dump()
P.log(f"=== SEALED-TEST SCORING tspulse done {time.strftime('%H:%M:%S')} ===  all done {time.time() - T0:.0f}s")
