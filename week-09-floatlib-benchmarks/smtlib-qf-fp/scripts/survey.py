#!/usr/bin/env python3
"""Survey the shapes of the SMT-LIB files of one family: comments stripped, literals replaced by placeholders."""
import os, re, sys, collections

root = sys.argv[1]
shapes = collections.Counter()
example = {}
for dirpath, _, files in os.walk(root):
    for name in sorted(files):
        if not name.endswith('.smt2'):
            continue
        path = os.path.join(dirpath, name)
        body = []
        with open(path, encoding='utf-8', errors='replace') as handle:
            for line in handle:
                line = line.split(';', 1)[0].strip() if not line.lstrip().startswith('(set-info :source') else '(set-info :source)'
                if line:
                    body.append(line)
        text = '\n'.join(body)
        text = re.sub(r'#b[01]+', '#b', text)
        text = re.sub(r'#x[0-9a-fA-F]+', '#x', text)
        text = re.sub(r'\(_ FloatingPoint \d+ \d+\)', '(_ FloatingPoint E S)', text)
        text = re.sub(r'\(_ bv\d+ \d+\)', '(_ bvN W)', text)
        text = re.sub(r'round[A-Za-z]+|RN[EA]|RT[PNZ]', 'RM', text)
        text = re.sub(r':status \w+', ':status ST', text)
        op = os.path.basename(dirpath)
        shapes[(op, text)] += 1
        example.setdefault((op, text), path)
for (op, text), count in sorted(shapes.items(), key=lambda item: (item[0][0], -item[1])):
    print(f'##### {op}: {count} files, e.g. {example[(op, text)]}')
    print(text)
