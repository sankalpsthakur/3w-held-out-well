"""Check submission/skab/evidence/report.json (written by the bench-events-skab-pkg kernel): printed metrics equal the expected rows and the pickle equals the frozen predictions row for row."""
import json
import sys
from pathlib import Path

r = json.loads((Path(__file__).resolve().parents[1] / "submission/skab/evidence/report.json").read_text())
ok = bool(r["ok"]) and r["row_check"]["mismatch_rows_total"] == 0 and r["row_check"]["rows_total"] == 23801 and r["printed_binary"] == r["expected_binary"] and all(r["changepoint_matches_expected"].values())
print("report.json ok:", ok, "| python", r["versions_imported"]["python"], "numpy", r["versions_imported"]["numpy"], "pandas", r["versions_imported"]["pandas"])
sys.exit(0 if ok else 1)
