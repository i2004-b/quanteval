# uses  saved baseline to produce metrics + a report row.
import json, csv, os
def write_json(d, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f: json.dump(d, f, indent=2)
def write_csv_row(row, out_path, header):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new: w.writeheader()
        w.writerow(row)

