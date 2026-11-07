# Real-world dataset registration

The first registered real-world source is UCI's
`ElectricityLoadDiagrams20112014` dataset. The dataset contains 15-minute
electricity consumption readings for 370 clients. DeepStoch does not commit the
large original download. Instead, provenance is stored in
`data/metadata/uci_electricity_load.json`, and the converter below extracts one
client into the canonical `timestamp,target` format.

Source: <https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112>

The UCI record identifies the dataset with DOI `10.24432/C58C86` and a CC BY
4.0 licence. Preserve that attribution in any external report or derivative
dataset. The UCI documentation also notes that readings are in kW and that
conversion to kWh requires division by four.

After downloading the source file from UCI, convert one client:

```powershell
python scripts/convert_uci_electricity.py `
  path\to\LD2011_2014.txt `
  data\real_world\uci_client_0.csv `
  --client MT_001 `
  --divide-by-four

python -m deepstoch validate-data data\real_world\uci_client_0.csv
python -m deepstoch compare data\real_world\uci_client_0.csv --test-size 672 --json
```

The exact client column must be read from the downloaded source header; `MT_001`
is an example placeholder, not a guaranteed column name. The raw source and
derived real-world data are excluded from the public code commit until their
storage and attribution policy is confirmed.

## Second registered source: Appliances Energy Prediction

DeepStoch also registers UCI's `Appliances Energy Prediction` dataset. The UCI
record describes 19,735 observations, 28 features, approximately 4.5 months of
10-minute measurements, and a CC BY 4.0 licence. Its DOI is `10.24432/C5VC8G`.
See `data/metadata/uci_appliances_energy.json` and the [UCI record](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction).

The intended canonical mapping is:

```text
date       -> timestamp
Appliances -> target
```

The converter now preserves selected sensor/weather columns as covariates:

```powershell
python scripts\convert_uci_appliances.py `
  path\to\energydata_complete.csv `
  data\real_world\uci_appliances.csv
```

The default covariates are `T1`, `RH_1`, `T2`, `RH_2`, `T_out`, `RH_out`,
`Windspeed`, and `Visibility`. Custom columns can be supplied with
`--covariates`. The dataset is registered now, but no
model result is claimed until the complete source file is downloaded, checked,
converted, validated, and evaluated chronologically.
