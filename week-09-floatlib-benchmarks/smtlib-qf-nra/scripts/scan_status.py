"""Scan the extracted QF_NRA benchmarks: status, family, size, declared symbols.

usage: python3 -I scan_status.py <root of extracted QF_NRA> <out.tsv>
Reads only; the benchmark files are data and are never executed.
"""
import os, re, sys, collections

root, out = sys.argv[1], sys.argv[2]
status_re = re.compile(rb"\(set-info\s+:status\s+(\w+)\s*\)")
decl_re = re.compile(rb"\(declare-(?:fun|const)\s+(\|[^|]*\||[^\s()]+)\s*(\(\s*\))?\s*(\w+)\s*\)")
rows = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames.sort()
    for name in sorted(filenames):
        if not name.endswith(".smt2"):
            continue
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, root)
        data = open(path, "rb").read()
        m = status_re.search(data)
        status = m.group(1).decode() if m else "none"
        sorts = collections.Counter(d.group(3).decode() for d in decl_re.finditer(data))
        rows.append((rel, rel.split(os.sep)[0], status, len(data), sorts.get("Real", 0), sorts.get("Bool", 0),
                     sum(v for k, v in sorts.items() if k not in ("Real", "Bool")),
                     data.count(b"(let "), data.count(b"(ite "), data.count(b"(define-fun")))
with open(out, "w") as f:
    f.write("path\tfamily\tstatus\tbytes\treals\tbools\tother_decls\tlets\tites\tdefine_funs\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
fam = collections.defaultdict(collections.Counter)
for r in rows:
    fam[r[1]][r[2]] += 1
print(f"{'family':50s} {'total':>6s} {'unsat':>6s} {'sat':>6s} {'unknown':>7s} other")
tot = collections.Counter()
for k in sorted(fam):
    c = fam[k]; tot.update(c)
    print(f"{k:50s} {sum(c.values()):6d} {c['unsat']:6d} {c['sat']:6d} {c['unknown']:7d} {sum(v for s, v in c.items() if s not in ('unsat','sat','unknown'))}")
print(f"{'all':50s} {sum(tot.values()):6d} {tot['unsat']:6d} {tot['sat']:6d} {tot['unknown']:7d}")
