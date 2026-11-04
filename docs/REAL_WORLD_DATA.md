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
