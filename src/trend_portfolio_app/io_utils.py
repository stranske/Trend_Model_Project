from __future__ import annotations
import os
import json
import datetime
import zipfile


def export_bundle(results, config_dict) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"trend_app_run_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    results.portfolio.to_csv(
        os.path.join(out_dir, "portfolio_returns.csv"), header=["return"]
    )
    ev = results.event_log_df()
    ev.to_csv(os.path.join(out_dir, "event_log.csv"))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results.summary(), f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, default=str)
    zip_path = out_dir + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_dir):
            for name in files:
                p = os.path.join(root, name)
                z.write(p, os.path.relpath(p, out_dir))
    return zip_path
