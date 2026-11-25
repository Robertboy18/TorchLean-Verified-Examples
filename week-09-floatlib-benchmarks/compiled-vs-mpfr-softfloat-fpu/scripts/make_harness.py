"""Write the comparison harness of this suite: FloatLib's script plus the public operations.

    python3 -I make_harness.py FLOATLIB_CHECKOUT OUT.sh

FloatLib's `benchmarks/scripts/format-comparison.sh` has no series for the public `Binary32` and
`Binary64` operations: its FloatLib series times the fixed-word kernels by name at 32 and 64 bits.
This script copies it and adds two series at 32 and 64 bits, which start the executable given in
`FORMAT_COMPARE_PUBLIC_BINARY` (built by `build_public.sh`) with the environment of the series
`binary-software` and another execution class: `binary-public` (the public operations, class
`proved-software-word-token`) and `binary-kernel` (the fixed-word kernels by name, class
`proved-kernel-word-token`). Every other line is FloatLib's: calibration, the shuffled order rotated
between trials, pinning, the validation of each row, the agreement check and the summary all run
unchanged on FloatLib's own series.

The copy differs from FloatLib's script in exactly these places (each is asserted to occur once):

 1. `root` is `FORMAT_COMPARE_ROOT` (the copy lives outside the checkout);
 2. the public executable and its output directory `public/`;
 3. `binary-public` and `binary-kernel` are accepted in `FORMAT_COMPARE_LANES`;
 4. they are available at 32 and 64 bits (`binary-public` at the widths of
    `FORMAT_COMPARE_PUBLIC_WIDTHS`, which is 32 and 64 unless set);
 5. they run the public executable with their execution class;
 6. their cells go to `environment/requested-matrix-public.csv`, not into the matrix that
    `verify_binary_agreement.py` checks (that script knows FloatLib's own series only);
 7. their rows go to `public/trial-NN.csv`, not into `raw/trial-NN.csv`, which FloatLib's
    verifier and plot read (two FloatLib rows in one cell are a duplicate for them);
 8. a failure of FloatLib's plot at the end does not discard the run.
"""
import sys
from pathlib import Path

checkout, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
text = (checkout / "benchmarks/scripts/format-comparison.sh").read_text()


def patch(old, new):
    global text
    assert text.count(old) == 1, (text.count(old), old)
    text = text.replace(old, new)


# 1
patch('root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"\n',
      'root="${FORMAT_COMPARE_ROOT:?the FloatLib checkout}"\n')
# 2
patch('softfloat_tree="$build_dir/softfloat-tree"\n',
      'softfloat_tree="$build_dir/softfloat-tree"\n'
      'public_benchmark="${FORMAT_COMPARE_PUBLIC_BINARY:-}"\n'
      'public_widths="${FORMAT_COMPARE_PUBLIC_WIDTHS:-32 64}"\n')
patch('  "$output/plots" \\\n  "$output/resources"\n',
      '  "$output/plots" \\\n  "$output/public" \\\n  "$output/resources"\n')
# 3
patch('''      posit-software|binary-software|binary-native-c|mpfr-reference)
        ;;
''', '''      posit-software|binary-software|binary-native-c|mpfr-reference)
        ;;
      binary-public|binary-kernel)
        if [[ ! -x "$public_benchmark" ]]; then
          echo "FORMAT_COMPARE_LANES requests $lane without FORMAT_COMPARE_PUBLIC_BINARY" >&2
          exit 2
        fi
        ;;
''')
# 4
patch('''    binary-native-c)
      [[ "$width" == 32 || "$width" == 64 ]]
      ;;
''', '''    binary-native-c)
      [[ "$width" == 32 || "$width" == 64 ]]
      ;;
    binary-public)
      [[ " $public_widths " == *" $width "* ]]
      ;;
    binary-kernel)
      [[ "$width" == 32 || "$width" == 64 ]]
      ;;
''')
# 5
patch('''    binary-native-c)
      command=(
''', '''    binary-public|binary-kernel)
      local public_class=proved-software-word-token
      if [[ "$lane" == binary-kernel ]]; then
        public_class=proved-kernel-word-token
      fi
      command=(
        env
        "FORMAT_COMPARE_WIDTH=$width"
        "FORMAT_COMPARE_OPERATION=$operation"
        "FORMAT_COMPARE_WARMUP_ITERATIONS=$warmup_iterations"
        "FORMAT_COMPARE_AGREEMENT_ITERATIONS=$agreement_iterations"
        ${iterations:+"FORMAT_COMPARE_ITERATIONS=$iterations"}
        "FORMAT_COMPARE_FAMILY=binary-interchange"
        "FORMAT_COMPARE_EXECUTION_CLASS=$public_class"
        "$public_benchmark"
      )
      ;;
    binary-native-c)
      command=(
''')
# 6
patch('''  for row in "${rows[@]}"; do
    printf '%s\\n' "${row//:/,}"
  done
} >"$requested_matrix"
''', '''  for row in "${rows[@]}"; do
    if [[ "$row" != binary-public:* && "$row" != binary-kernel:* ]]; then
      printf '%s\\n' "${row//:/,}"
    fi
  done
} >"$requested_matrix"
{
  printf '%s\\n' "lane,totalBits,operation"
  for row in "${rows[@]}"; do
    if [[ "$row" == binary-public:* || "$row" == binary-kernel:* ]]; then
      printf '%s\\n' "${row//:/,}"
    fi
  done
} >"$output/environment/requested-matrix-public.csv"
''')
# 7
patch('''    if [[ "$first_row" == 1 ]]; then
      cp "$row_csv" "$trial_csv"
      first_row=0
    else
''', '''    if [[ "$lane" == binary-public || "$lane" == binary-kernel ]]; then
      public_csv="$output/public/trial-$trial_name.csv"
      if [[ -e "$public_csv" ]]; then
        tail -n 1 "$row_csv" >>"$public_csv"
      else
        cp "$row_csv" "$public_csv"
      fi
    elif [[ "$first_row" == 1 ]]; then
      cp "$row_csv" "$trial_csv"
      first_row=0
    else
''')
# 8
patch('''python3 "$root/benchmarks/plots/format_comparison.py" \\
  "$output/raw" "$output/plots"
''', '''python3 "$root/benchmarks/plots/format_comparison.py" \\
  "$output/raw" "$output/plots" ||
  echo "the plot failed; the raw rows are complete" >&2
''')
out.write_text(text)
out.chmod(0o755)
print(out)
