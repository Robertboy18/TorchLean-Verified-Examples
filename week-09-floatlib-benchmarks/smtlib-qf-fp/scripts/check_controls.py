#!/usr/bin/env python3
"""check_controls.py: the outcome of the control batches.

* Opposite claims (`#sat` or `#unsat` of an instance whose status is known): none may be accepted.
* Mutated claims (`#mutated`): accepted only when the mutated literal is a NaN, which the flip leaves a NaN.

Usage: check_controls.py SIDE.csv ROOT
"""
import csv, os, re, sys

side, root = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(side, newline='', encoding='utf-8')))
opposite = mutated = opposite_accepted = mutated_accepted = mutated_nan = 0
unexpected = []
for row in rows:
    theorem = row['theorem']
    if theorem.endswith('#mutated»'):
        mutated += 1
        if row['proved'] == '1':
            mutated_accepted += 1
            text = open(os.path.join(root, row['instance']), encoding='utf-8').read()
            literal = re.findall(r'\(assert \(= r (.*)\)\)\n', text)[-1]
            is_nan = literal.startswith('(_ NaN')
            match = re.match(r'\(fp #b([01]) #b([01]+) #(b|x)([0-9a-fA-F]+)\)', literal)
            if match:
                fraction = int(match.group(4), 2 if match.group(3) == 'b' else 16)
                is_nan = set(match.group(2)) == {'1'} and fraction != 0
            if is_nan:
                mutated_nan += 1
            else:
                unexpected.append((row['instance'], literal))
    elif row['status'] in ('sat', 'unsat') and row['claims'] != row['status']:
        opposite += 1
        if row['proved'] == '1':
            opposite_accepted += 1
            unexpected.append((row['instance'], 'opposite accepted'))
print(f'opposite claims: {opposite}, accepted {opposite_accepted}')
print(f'mutated claims: {mutated}, accepted {mutated_accepted}, of which the expected result is a NaN: {mutated_nan}')
print('unexpected:', unexpected)
