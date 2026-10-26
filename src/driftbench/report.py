import json, sys

def main(path):
    d = json.load(open(path))
    print("config:", d["config"]); print("default rules FAR:", d["default_rules_far"])
    print("thresholds:", {k: round(v, 4) for k, v in d["thresholds"].items()})
    dets = list(d["thresholds"])
    print("\nnull stream false-alarm rate (target 0.05):", {c["det"]: c["far"] for c in d["cells"] if c["kind"] == "null"})
    print("\npower (delay in batches, median) per shift:")
    print("%-14s" % "shift" + "".join("%-18s" % x for x in dets))
    keys = []
    for c in d["cells"]:
        if c["kind"] != "null" and (c["kind"], c["mag"]) not in keys: keys.append((c["kind"], c["mag"]))
    for k in keys:
        row = "%-14s" % ("%s %.2f" % k)
        for x in dets:
            c = next(c for c in d["cells"] if c["det"] == x and (c["kind"], c["mag"]) == k)
            row += "%-18s" % ("%.2f (%s)" % (c["power"], "-" if c["median_delay"] is None else "%g" % c["median_delay"]))
        print(row)

if __name__ == "__main__":
    main(sys.argv[1])
