# uses  saved baseline to produce metrics + a report row.
import json, csv, os, datetime

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

def add_timestamp(d):
    d["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    return d

def log_experiment(metrics, model_name, dataset, method, out_dir="outputs/reports"):
    """
    metrics: dict of metric_name -> value
    model_name: str ("ResNet18" / "DistilBERT")
    dataset: str ("CIFAR-10" / "SST-2")
    method: str ("baseline" / "qat" / "ptq" / "adapter")
    """
    os.makedirs(out_dir, exist_ok=True)

    # add metadata + timestamp
    record = {
        "model": model_name,
        "dataset": dataset,
        "method": method,
        **metrics
    }
    record = add_timestamp(record)

    # save JSON (per run)
    json_path = os.path.join(out_dir, f"{model_name}_{dataset}_{method}.json")
    write_json(record, json_path)

    # append CSV row (all runs)
    csv_path = os.path.join(out_dir, "all_results.csv")
    header = list(record.keys())
    write_csv_row(record, csv_path, header)

    return record

def read_results(csv_path="outputs/reports/all_results.csv"):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)