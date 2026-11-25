#!/bin/bash
# lcrun.sh ARGS...: run `lake env ARGS` inside the ported LeanCert checkout (`fetch.sh` makes it).
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
cd "${LEANCERT_DIR:-$suite/work/leancert}" || exit 2
exec lake env "$@"
