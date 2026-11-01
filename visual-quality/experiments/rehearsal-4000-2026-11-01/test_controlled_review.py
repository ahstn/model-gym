"""Check native crop alignment, information isolation and complete comparison review."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from controlled_review import parent_crop_box, render_family, review_template, sha, validate


class ControlledReviewTests(unittest.TestCase):
    def fixture(self, directory):
        rows = []
        for index in range(4):
            pixels = np.random.default_rng(index).integers(0, 256, (270, 400, 3), dtype=np.uint8)
            path = directory / f'{index}.png'
            Image.fromarray(pixels).save(path)
            rows.append({'case_id': f'R{index:04d}', 'image_id': f'image-{index}', 'family_id': 'family',
                'image_path': str(path), 'source_path': str(path), 'view_sha256': sha(path), 'source_sha256': sha(path),
                'role': 'source_parent' if index == 0 else 'variant', 'severity': None if index == 0 else index,
                'grader_score': 4.9, 'reference_score': 2.1, 'stats_text': 'secret unwanted annotation',
                'transform': {'secret_parameter': 99}})
        return rows

    def test_crops_are_exact_parent_selected_pixels_for_all_members(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows = self.fixture(root)
            family = render_family(rows, 1, root / 'assets')
            with Image.open(rows[0]['image_path']) as parent:
                expected_box = parent_crop_box(parent)
            for row, member in zip(rows, family['members']):
                self.assertEqual(member['crop_box_xyxy'], list(expected_box))
                with Image.open(row['image_path']) as original, Image.open(member['crop_path']) as crop:
                    np.testing.assert_array_equal(np.asarray(crop), np.asarray(original.crop(expected_box)))
            serialized = json.dumps(family)
            self.assertNotIn('grader_score', serialized)
            self.assertNotIn('reference_score', serialized)
            self.assertNotIn('secret_parameter', serialized)
            self.assertNotIn('unwanted annotation', serialized)

    def test_template_relations_follow_parent_and_immediate_previous(self):
        members = [{'case_id': f'R{i}', 'view_sha256': str(i)} for i in range(4)]
        family = {'members': members, 'family_id': 'family', 'review_family_id': 'F001', 'contact_sheet_sha256': 'sheet'}
        comparisons = review_template(family)['comparisons']
        self.assertEqual([r['versus_parent']['comparison_case_id'] for r in comparisons], ['R0'] * 3)
        self.assertEqual([r['versus_previous']['comparison_case_id'] for r in comparisons], ['R0', 'R1', 'R2'])

    def test_validation_requires_all_families_and_consistent_v1(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            asset = root / 'validation-fixture.bin'
            asset.write_bytes(b'identity-check-fixture')
            families = [{'members': [{'case_id': f'R{f}-{i}', 'view_sha256': sha(asset),
                                      'image_path': str(asset), 'crop_path': str(asset), 'crop_sha256': sha(asset)} for i in range(4)],
                         'family_id': f'family-{f}', 'review_family_id': f'F{f:03d}',
                         'contact_sheet_path': str(asset), 'contact_sheet_sha256': sha(asset)} for f in range(10)]
            packet = root / 'packet.json'
            packet.write_text(json.dumps({'families': families}))
            review = {'packet_sha256': sha(packet), 'reviewer': 'unit-test-only', 'families': [review_template(f) for f in families]}
            for family in review['families']:
                family['reviewed_full_frames_and_native_crops'] = True
                for item in family['comparisons']:
                    for relation in ('versus_parent', 'versus_previous'):
                        item[relation].update(classification='not_clearly_worse', evidence='Fixture decision for validator test.')
            output = root / 'reviews.json'
            output.write_text(json.dumps(review))
            validate(packet, output)
            broken = copy.deepcopy(review)
            broken['families'][0]['comparisons'][0]['versus_previous']['classification'] = 'clear_degradation'
            output.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                validate(packet, output)
            broken = copy.deepcopy(review)
            broken['families'].pop()
            output.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                validate(packet, output)


if __name__ == '__main__':
    unittest.main()
