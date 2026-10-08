# Kernel entry: build the benchmark kit tables (decision grid with eligibility) for the 1,119 real 3W instances.
import csv, hashlib, io, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow.parquet as pq
from w3 import scorer

COMMIT = "d57c058af6baf65f1b285f3e56f46e4f018a07d9"
OUT = "/kaggle/working"
index = json.load(open("/tmp/w3code/data/index_labels.json"))
files = sorted(index["files"], key=lambda f: f["path"])
t0 = time.time()

def fetch(path):
    url = f"https://raw.githubusercontent.com/petrobras/3W/{COMMIT}/{path}"
    last = None
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "bench-events-kit/1"}), timeout=120) as r:
                return r.read()
        except Exception as e:
            last = e; time.sleep(1.5 * (k + 1))
    raise last

def one(f):
    d = fetch(f["path"])
    blob = hashlib.sha1(b"blob " + str(len(d)).encode() + b"\0" + d).hexdigest()
    assert blob == f["sha"], f["path"]
    tbl = pq.read_table(io.BytesIO(d), columns=[c for c in scorer.PROCESS_CHANNELS if c in pq.ParquetFile(io.BytesIO(d)).schema_arrow.names])
    cols = [np.asarray(tbl[c].to_numpy(zero_copy_only=False), dtype=np.float64) for c in tbl.column_names]
    any_proc = np.isfinite(np.column_stack(cols)).any(axis=1) if cols else np.zeros(f["rows"], dtype=bool)
    assert len(any_proc) == f["rows"], f["path"]
    assert abs(any_proc.mean() - f["any_process_finite_frac"]) < 1e-9, (f["path"], any_proc.mean(), f["any_process_finite_frac"])
    return scorer.eligibility_from_any_process(any_proc)

with ThreadPoolExecutor(8) as ex:
    elig_list = list(ex.map(one, files))
elig_by_path = {f["path"]: e for f, e in zip(files, elig_list)}
inst, r, y, el, st = scorer.make_decision_table(files, elig_by_path)
np.savez_compressed(f"{OUT}/decisions.npz", inst=inst, r=r, y=y, elig=el, state=st)
with open(f"{OUT}/instances.csv", "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["idx", "path", "well", "folder_class", "rows", "sha"])
    for i, f in enumerate(files):
        w.writerow([i, f["path"], f["well"], f["folder_class"], f["rows"], f["sha"]])
summary = {"n_instances": len(files), "n_decisions": int(len(inst)), "eligible_decisions": int(el.sum()), "labelled_decisions": int((y >= 0).sum()),
           "labelled_eligible_decisions": int(((y >= 0) & el).sum()), "seconds": round(time.time() - t0, 1)}
json.dump(summary, open(f"{OUT}/kit_summary.json", "w"), indent=1)
print(json.dumps(summary))
