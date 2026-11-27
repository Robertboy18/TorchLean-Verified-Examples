#!/bin/bash
# flrun.sh ARGS...: run `lake env ARGS` in this week's Lake project, where `lean` sees FloatLib.
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
cd "${FLOATLIB_PROJECT:-$week}" || exit 2
exec lake env "$@"
