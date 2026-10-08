"""Write kernels/skab_input/EZH.ipynb (unexecuted source; the executed PR copy is submission/skab/EZH.ipynb) in the layout of SKAB's notebooks/Conv_AE.ipynb.

Conventions copied from the SKAB repository: nbformat 4.4, empty kernelspec, `# [donotremove]` on the cells whose outputs are kept by the
repository's pre-commit notebook cleanup, 79-column black style. The executed copy is produced by kernels/skab_pkg.py in SKAB's pinned
environment.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    md("# Pipeline for the anomaly detection on the SKAB using EZH (EWMA-z score with hysteresis)"),
    md("Details regarding the SKAB one can find in the [SKAB repository](https://github.com/waico/SKAB)."),
    code(
        "# libraries importing\n"
        "import sys\n"
        "import warnings\n"
        "\n"
        "import pandas as pd\n"
        "\n"
        'sys.path.append("..")\n'
        "from core.EZH import EZH\n"
        "from core.metrics import chp_score\n"
        "from core.utils import load_preprocess_skab, plot_results\n"
        "\n"
        'warnings.filterwarnings("ignore", category=UserWarning)'
    ),
    md("## Data"),
    code("Xy_traintest_list = load_preprocess_skab()"),
    md("## Method"),
    md("### Method initialization"),
    code("# model defining\nmodel = EZH()"),
    md("### Method fitting and applying"),
    code(
        "# inference\n"
        "predicted_outlier, predicted_cp = [], []\n"
        "true_outlier, true_cp = [], []\n"
        "for X_train, X_test, y_train, y_test in Xy_traintest_list:\n"
        "    # model fitting on the anomaly-free training part\n"
        "    model.fit(X_train.values)\n"
        "\n"
        "    # results predicting (causal: row by row)\n"
        "    prediction = pd.Series(\n"
        "        model.predict(X_test.values).astype(int), index=X_test.index\n"
        "    )\n"
        "\n"
        "    # predicted outliers saving\n"
        "    predicted_outlier.append(prediction)\n"
        "\n"
        "    # predicted CPs saving\n"
        "    prediction_cp = abs(prediction.diff())\n"
        "    prediction_cp.iloc[0] = prediction.iloc[0]\n"
        "    predicted_cp.append(prediction_cp)\n"
        "\n"
        '    true_outlier.append(y_test["anomaly"])\n'
        '    true_cp.append(y_test["changepoint"])'
    ),
    md("### Results visualization"),
    code("# [donotremove]\nplot_results(\n    (true_outlier[1], predicted_outlier[1]),\n    (true_cp[1], predicted_cp[1]),\n)"),
    md("## Evaluation (metrics calculation)"),
    code(
        "import pickle\n"
        "\n"
        "pickle.dump(\n"
        "    predicted_outlier,\n"
        '    open(f"../results/results-{model.__class__.__name__}.pkl", "wb"),\n'
        ")"
    ),
    md("### Binary classification (outlier detection) metrics"),
    code(
        "# [donotremove]\n"
        "# binary classification metrics calculation\n"
        'binary = chp_score(true_outlier, predicted_outlier, metric="binary")'
    ),
    md("### Changepoint detection metrics"),
    code(
        "# [donotremove]\n"
        "# average detection delay metric calculation\n"
        "add = chp_score(\n"
        "    true_cp,\n"
        "    predicted_cp,\n"
        '    metric="average_time",\n'
        '    window_width="60s",\n'
        '    anomaly_window_destination="righter",\n'
        ")"
    ),
    code(
        "# [donotremove]\n"
        "# nab metric calculation\n"
        "nab = chp_score(\n"
        "    true_cp,\n"
        "    predicted_cp,\n"
        '    metric="nab",\n'
        '    window_width="60s",\n'
        '    anomaly_window_destination="righter",\n'
        ")"
    ),
]
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "", "language": "python", "name": ""},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.14",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = ROOT / "kernels" / "skab_input" / "EZH.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", out, len(cells), "cells")
