import sys

from mjo.data import load_bom_rmm, load_city_temperature
from mjo.features import (
    compute_daily_anomalies,
    merge_temp_with_rmm,
    add_autoregressive_features,
    build_lead_frame,
)
from mjo.models import fit_all_leads


def run(city_name, temp_file):

    print(f"\nRunning MJO forecast pipeline for: {city_name}\n")

    rmm = load_bom_rmm("data/rmm.74toRealtime.txt")

    temp = load_city_temperature(temp_file)
    temp = compute_daily_anomalies(temp)

    merged = merge_temp_with_rmm(temp, rmm)
    merged = add_autoregressive_features(merged)

    df_leads = build_lead_frame(merged, max_lead=20)

    models = fit_all_leads(df_leads, max_lead=20)

    print(f"✅ Finished {city_name}")
    print(f"Models built for leads: {list(models.keys())}")

    return models


if __name__ == "__main__":

    city = sys.argv[1]
    file = sys.argv[2]

    run(city, file)