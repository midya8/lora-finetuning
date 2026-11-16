#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps "$root" --wheel-dir "$tmp/wheels" >/dev/null
python -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --no-index "$tmp"/wheels/*.whl >/dev/null
# Copy tests, not implementation files, so subprocess imports use the wheel too.
cp -R "$root/tests" "$tmp/tests"
find "$tmp/tests" -type d -name __pycache__ -prune -exec rm -rf {} +
# One original test reads this exact fixture by checkout-relative path. Keep only
# its inert data, never a second importable copy of the package implementation.
mkdir -p "$tmp/loralab/rank_data"
cp "$root/loralab/rank_data/9-rank-witnesses.json" "$tmp/loralab/rank_data/"
cd "$tmp"
"$tmp/venv/bin/python" - <<'CHECK'
import importlib
import pathlib
import sys
import unittest

modules = [
    'loralab', 'loralab.cli', 'loralab.data', 'loralab.eval', 'loralab.lora',
    'loralab.model', 'loralab.coordinates', 'loralab.coordinate_fixtures',
    'loralab.coordinate_report', 'loralab.predictive_rank',
    'loralab.rank_fixtures', 'loralab.rank_verify',
]
prefix = pathlib.Path(sys.prefix).resolve()

def check_origins():
    for name in modules:
        origin = pathlib.Path(importlib.import_module(name).__file__).resolve()
        assert prefix in origin.parents, (name, origin, prefix)
    print(f'All {len(modules)} module origins are inside the wheel environment',
          flush=True)

check_origins()
suite = unittest.defaultTestLoader.discover('tests', top_level_dir='.')
count = suite.countTestCases()
assert count > 0, 'No installed tests were discovered'
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.testsRun == count, (result.testsRun, count)
assert result.wasSuccessful(), 'Installed suite failed'
assert not result.skipped, result.skipped
check_origins()
print(f'All {count} unchanged Python tests passed outside the checkout', flush=True)
CHECK
"$tmp/venv/bin/python" -m loralab evaluate --format json > evaluate.json
"$tmp/venv/bin/python" -m loralab trajectory > trajectory.json
"$tmp/venv/bin/python" -m loralab gates --min-base-drop .3 --min-adapted-accuracy .95 --max-rank2-param-share .7 --max-merge-diff 1e-12 > gates.txt
"$tmp/venv/bin/python" - <<'PY'
import gzip,importlib,json,math,pathlib
from importlib.resources import files
for name in ['cli','data','eval','lora','model','coordinates','coordinate_fixtures','coordinate_report']:
    importlib.import_module('loralab.'+name)
from loralab.coordinate_fixtures import build_archive,problem_at,trajectory_report
from loralab.coordinates import audit
root=files('loralab').joinpath('coordinate_data')
archive=json.loads(gzip.decompress(root.joinpath('6-original-epochs.json.gz').read_bytes()))
regenerated=build_archive()
def compare(a,b,path='root',tol=1e-11):
    if isinstance(b,dict):
        assert a.keys()==b.keys(),path
        for k in b:compare(a[k],b[k],path+'.'+k,tol)
    elif isinstance(b,list):
        assert len(a)==len(b),path
        for i,v in enumerate(b):compare(a[i],v,path+'['+str(i)+']',tol)
    elif isinstance(b,float):assert math.isfinite(a) and math.isclose(a,b,rel_tol=tol,abs_tol=tol),(path,a,b)
    else:assert a==b,(path,a,b)
compare(regenerated,archive)
assert json.load(open('evaluate.json'))==regenerated['original_report']
compare(json.load(open('trajectory.json')),trajectory_report(archive),tol=1e-12)
for item in json.loads(gzip.decompress(root.joinpath('7-witnesses.json.gz').read_bytes())):
    s=item['selection']
    compare(trajectory_report(archive,s['rank'],s['epoch'],s['batch'],s['scale'],s['learning_rate']),item['report'],tol=1e-12)
for item in json.loads(gzip.decompress(root.joinpath('8-controls.json.gz').read_bytes())):
    compare(audit(item['problem'],**item['options']),item['report'],tol=1e-12)
assert sum(len(r['states'])for r in archive['runs'])==124
assert 'All gates passed' in pathlib.Path('gates.txt').read_text()
pathlib.Path('input.json').write_text(json.dumps(problem_at(archive)))
print('Eight implementation modules import; full original data/base/factors/124states/report regenerated at1e-11; same-runtime report exact;54complete frozen reports at1e-12 outside checkout')
PY
"$tmp/venv/bin/python" -m loralab coordinates --input input.json > coordinates.json
"$tmp/venv/bin/python" - <<'PY'
import json
r=json.load(open('coordinates.json'))
assert r['schema']=='loralab.factor-coordinates.v1'
assert r['same_model_logits_max_abs_residual']<1e-12
assert r['compensated_next_delta_max_abs_residual']<1e-12
print('All four CLI commands execute from isolated installed wheel')
PY
