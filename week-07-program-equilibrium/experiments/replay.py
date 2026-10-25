"""Replay the exported Inkling results, optionally rechecking certificates in Lean.

No API access, model weights or third-party Python packages are needed. This checks
the evaluated programs and reported scores, not the private training trajectory.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def load():
    """Reconstruct the original requests; pooling changes storage, not their hashes."""
    data = json.loads((HERE / "inkling.json").read_text())
    require(data["format"] == "program-equilibrium-inkling-v1", "unknown export format")
    requests, expected = {}, {}
    for shard in data["checks"]:
        raw = (HERE / shard["file"]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == shard["sha256"], "changed certificate file")
        part = json.loads(raw)
        require(not (requests.keys() & part["certificates"].keys()), "duplicate certificate")
        requests.update(part["certificates"])
        expected.update(part["expected"])
    for key, schema in data["schemas"].items():
        require(digest(schema) == key, "changed schema")
    for key, population in data["populations"].items():
        require(digest(population) == key, "changed population")
    require(requests.keys() == expected.keys(), "missing verdict")
    for key, stored in list(requests.items()):
        population = data["populations"][stored["population"]]
        payload = {k: v for k, v in stored.items() if k != "population"}
        payload["schemas"] = [data["schemas"][schema] for schema in population]
        require(digest(payload) == key, "changed request")
        requests[key] = payload
        verdict = expected[key]
        require(verdict["valid_certificate"] and verdict["stopped"] and verdict["table_matches"],
                "export contains an uncertified request")
        table, a = payload["table"], payload["candidate"]
        require(len(table) == len(population) and all(len(row) == len(table) for row in table),
                "malformed table")
        require(all(type(value) is bool for row in table for value in row), "non-Boolean move")
        safe = all(not table[a][b] or table[b][a] for b in range(len(table)))
        mutual = sum(table[a][b] and table[b][a] for b in range(len(table)) if b != a)
        require(verdict["safe"] == safe and verdict["external_cooperation_count"] == mutual,
                "verdict disagrees with outcome table")
        require(verdict["self_cooperation"] == table[a][a] and
                verdict["accepted"] == (safe and table[a][a] and mutual > 0),
                "incorrect admission decision")
        require(verdict["candidate"] == a and verdict["world"] == payload["world"],
                "verdict belongs to another request")
    return data, requests, expected


def score(key, requests, expected):
    verdict = expected[key]
    return (verdict["external_cooperation_count"] / (len(requests[key]["schemas"]) - 1)
            if verdict["accepted"] else 0.0)


def close(actual, claimed, label):
    require(math.isclose(actual, claimed, rel_tol=1e-12, abs_tol=1e-12), label)


def summarize(data, requests, expected):
    """Keep rejected/invalid attempts in denominators, and transfer attempt weights intact."""
    for name, study in data["studies"].items():
        print(f"\n{name}: {study['optimizer_updates']} optimizer updates")
        for arm in ("unrestricted", "shielded"):
            results = []
            for phase in ("baseline", "final"):
                rows = [row for seed in study["seeds"]
                        for row in study["evaluations"][f"{arm}-{seed}-{phase}"]]
                require(len(rows) == 128, "wrong evaluation denominator")
                keys = [row["certificate"] for row in rows if "certificate" in row]
                admitted = sum(expected[key]["accepted"] for key in keys) / len(rows)
                cooperation = sum(score(key, requests, expected) for key in keys) / len(rows)
                claim = study["arms"][arm]["all_test_cases"][phase]
                close(admitted, claim["admitted_fraction"], "changed admission rate")
                close(cooperation, claim["cooperation_per_attempt"], "changed cooperation rate")
                close(len(keys) / len(rows), claim["protocol_valid_fraction"], "changed syntax rate")
                close(sum(not expected[key]["safe"] for key in keys) / len(rows),
                      claim["exploitable_proposal_fraction"], "changed exploitation rate")
                results.append(f"{admitted:.2%} admitted, {cooperation:.2%} cooperation")
            print(f"  {arm}: {results[0]} -> {results[1]}")
        transfer = study["transfer"]
        checks = {(row["schema_sha256"], row["case_id"]): row["certificate"]
                  for row in transfer["checks"]}
        require(len(checks) == len(transfer["checks"]), "duplicate transfer computation")
        cases = sorted({case for _, case in checks})
        require(len(cases) == 8, "wrong transfer population count")
        for (schema, _), key in checks.items():
            payload = requests[key]
            require(digest(payload["schemas"][payload["candidate"]]) == schema,
                    "transfer changed the candidate")
        for arm in ("unrestricted", "shielded"):
            cohort = [row for row in transfer["cohort"] if row["arm"] == arm]
            counts = Counter()
            for row in cohort:
                original = requests[row["original_payload_sha256"]]
                require(digest(original["schemas"][original["candidate"]]) == row["schema_sha256"],
                        "transfer is not the original candidate")
                require(expected[row["original_payload_sha256"]]["accepted"],
                        "transfer included an originally rejected proposal")
                for case in cases:
                    verdict = expected[checks[row["schema_sha256"], case]]
                    counts["certified"] += verdict["valid_certificate"]
                    counts["readmitted"] += verdict["accepted"]
                    counts["certified_exploitable"] += not verdict["safe"]
            claim = transfer["counts"][arm]
            require(len(cohort) == claim["originally_admitted_attempts"], "changed transfer cohort")
            require(len(cohort) * len(cases) == claim["weighted_transfer_attempts"],
                    "changed transfer weights")
            for key in counts:
                require(counts[key] == claim[key], f"changed transfer {key}")
            print(f"  transfer {arm}: {counts['readmitted']}/{counts['certified']} readmitted; "
                  f"{counts['certified_exploitable']} exploitable, all rejected")
    print("\nhandwritten controls, original 32 populations:")
    for name in sorted({row["name"] for row in data["controls"]}):
        keys = [row["certificate"] for row in data["controls"] if row["name"] == name]
        require(len(keys) == 32, "wrong control denominator")
        print(f"  {name}: {sum(score(key, requests, expected) for key in keys) / len(keys):.2%}")
    search = data["delayed_search"]
    require(len(search["proposals"]) == 48, "wrong search proposal count")
    for label, report in search["reports"].items():
        keys = [row["certificate"] for row in report["evaluations"]]
        for key in keys:
            payload = requests[key]
            require(payload["schemas"][payload["candidate"]] == report["schema"],
                    f"wrong policy in {label}")
        cooperation = sum(score(key, requests, expected) for key in keys) / len(keys)
        close(cooperation, report["cooperation"], f"changed score in {label}")
    print("\ndelayed-guard follow-up, 48 held-out populations:")
    for label, report in search["reports"].items():
        if label.startswith(("heldout-", "secondary-")):
            require(len(report["evaluations"]) == 48, "wrong held-out denominator")
            print(f"  {label}: {report['cooperation']:.2%}")


def recheck(keys, requests, expected):
    """One fixed Lean process reads data only; it never executes model-generated source."""
    command = ["lake", "--no-build", "exe", "check_program_policy"]
    process = subprocess.Popen(command, cwd=ROOT, text=True, bufsize=1,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=(os.name == "posix"))
    try:
        def send(payload):
            process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                require(bool(selector.select(60)), "Lean timed out: no verdict obtained")
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("Lean stopped before returning a verdict: " +
                                   process.stderr.read(4096))
            return json.loads(line)

        for index, key in enumerate(keys, 1):
            actual = send(requests[key])
            require(actual == {**expected[key], "table": requests[key]["table"]},
                    f"Lean verdict changed for {key}")
            if len(keys) <= 3 or index % 100 == 0:
                print(f"  Lean {index}/{len(keys)}: {key[:12]} accepted={actual['accepted']}", flush=True)
            if len(keys) <= 3:
                a, table = actual["candidate"], actual["table"]
                exploiters = [b for b in range(len(table)) if table[a][b] and not table[b][a]]
                print(f"    settled at world {actual['world']}; exploiting opponents: {exploiters}")
        # A correct outcome table is part of the certificate, not an unchecked annotation.
        bad = json.loads(json.dumps(requests[keys[0]]))
        bad["table"][0][0] = not bad["table"][0][0]
        verdict = send(bad)
        require(not verdict["valid_certificate"] and not verdict["accepted"],
                "Lean accepted a corrupted table")
        process.stdin.close()
        require(process.wait(timeout=10) == 0, "Lean exited unsuccessfully")
        print(f"Lean matched all {len(keys)} verdicts and rejected the corrupted table.")
    finally:
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
        process.stdout.close()
        process.stderr.close()
        if not process.stdin.closed:
            process.stdin.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", choices=("example", "all"),
                        help="recheck the transfer counterexample, or every exported certificate")
    args = parser.parse_args()
    data, requests, expected = load()
    summarize(data, requests, expected)
    print(f"\nBound {len(requests)} distinct requests to their exported hashes.")
    if args.lean:
        keys = (list(data["examples"]["transfer"].values()) if args.lean == "example"
                else sorted(requests))
        recheck(keys, requests, expected)
    else:
        print("Summary replay only. Use --lean example or --lean all to recompute modal outcomes.")


if __name__ == "__main__":
    main()
