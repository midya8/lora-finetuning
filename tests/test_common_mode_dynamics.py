"""Finite regression checks for a class-common shift in the output factor.

B -> B + 1 v^T changes raw logits, but not the softmax objective. Because
class errors sum to zero, ordinary factor SGD preserves this relation too.
This is not the general invertible factor gauge tested in test_coordinates.
"""
import copy
import gzip
import json
import math
import random
import unittest
from importlib.resources import files

from loralab.coordinate_fixtures import problem_at
from loralab.coordinates import audit, matmul
from loralab.eval import build_fixture
from loralab.lora import LoRAClassifier
from tests.test_coordinates import problem


def shifted(p, offset):
    out = copy.deepcopy(p)
    out['B'] = [[value + shift for value, shift in zip(row, offset)]
                for row in p['B']]
    return out


class CommonModeDynamicsTests(unittest.TestCase):
    def close(self, actual, expected):
        if isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.close(a, b)
        else:
            self.assertTrue(math.isclose(actual, expected, rel_tol=2e-10,
                                         abs_tol=2e-11),
                            f'{actual!r} != {expected!r}')

    def check_step(self, p, offset, rate):
        q = shifted(p, offset)
        original = copy.deepcopy(p)
        changed = copy.deepcopy(q)
        a = audit(p, learning_rate=rate)['original']
        b = audit(q, learning_rate=rate)['original']
        self.assertEqual(p, original)
        self.assertEqual(q, changed)
        for key in ['gradient_A', 'gradient_B', 'A_after']:
            self.close(b[key], a[key])
        self.close(b['B_after'], [[value + shift
                    for value, shift in zip(row, offset)]
                    for row in a['B_after']])
        for stage, factors in [('before', p['A']), ('after', a['A_after'])]:
            self.close(b[stage]['loss'], a[stage]['loss'])
            self.close(b[stage]['probabilities'], a[stage]['probabilities'])
            self.close(b[stage]['gradient_delta'], a[stage]['gradient_delta'])
            # The common offset must explain every raw logit, not only loss.
            common = matmul([offset], factors)[0]
            for example, left, right in zip(p['examples'],
                    a[stage]['logits'], b[stage]['logits']):
                shift = math.fsum(x * y for x, y in zip(common, example['x']))
                self.close(right, [value + shift for value in left])
            self.close(b[stage]['delta'], [[value + shift
                        for value, shift in zip(row, common)]
                        for row in a[stage]['delta']])
        return a, b

    def test_all_archived_states_and_selected_full_batches(self):
        archive = json.loads(gzip.decompress(files('loralab').joinpath(
            'coordinate_data/6-original-epochs.json.gz').read_bytes()))
        count = 0
        for run in archive['runs']:
            width = run['factor_width']
            offset = [(-1)**j * (j + 1) / 16 for j in range(width)]
            for state in run['states']:
                epoch = state['epoch']
                for rate in [0, .1]:
                    self.check_step(problem_at(archive, width, epoch), offset, rate)
                    count += 1
                if epoch in [0, 1, 30]:
                    self.check_step(problem_at(archive, width, epoch, 'full_train'),
                                    offset, .05)
        self.assertEqual(count, 248)

    def test_generated_shapes_and_multistep_dynamics(self):
        for seed in range(36):
            p = problem(seed, m=2 + seed % 7, d=1 + seed % 6, r=1 + seed % 8)
            offset = [(-1)**j * .125 for j in range(len(p['A']))]
            # Reuse each updated problem so this checks a trajectory, not only
            # repeated evaluations at the starting point.
            for rate in [0, .03125, .5]:
                a, _ = self.check_step(p, offset, rate)
                p['A'], p['B'] = a['A_after'], a['B_after']

    def test_original_online_trainer_preserves_common_mode(self):
        fx = build_fixture()
        for width in [1, 2, 4, 8]:
            left = LoRAClassifier(fx['base'], width, random.Random(0))
            right = copy.deepcopy(left)
            offset = [(-1)**j / 8 for j in range(width)]
            right.B = [[value + shift for value, shift in zip(row, offset)]
                       for row in right.B]
            # Check after every actual online update, including zero-B start.
            for x, y in fx['trainB'][:24]:
                left.train_epoch([(x, y)], fx['lr'])
                right.train_epoch([(x, y)], fx['lr'])
                self.close(right.A, left.A)
                self.close(right.B, [[value + shift
                           for value, shift in zip(row, offset)] for row in left.B])
                common = math.fsum(v * math.fsum(a * t for a, t in zip(row, x))
                                   for v, row in zip(offset, left.A))
                self.close(right.logits(x), [z + common for z in left.logits(x)])

    def test_noncommon_shift_and_raw_delta_are_not_invariants(self):
        p = problem(5)
        a, b = self.check_step(p, [.25, -.125], .1)
        self.assertGreater(max(abs(x-y) for left, right in
            zip(a['before']['delta'], b['before']['delta'])
            for x, y in zip(left, right)), .01)
        q = copy.deepcopy(p)
        q['B'][0][0] += .5
        wrong = audit(q, learning_rate=.1)['original']
        self.assertGreater(abs(wrong['before']['loss'] - a['before']['loss']), 1e-5)
        self.assertGreater(max(abs(x-y) for left, right in
            zip(wrong['gradient_A'], a['gradient_A'])
            for x, y in zip(left, right)), 1e-5)


if __name__ == '__main__':
    unittest.main()
