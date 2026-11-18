from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests import test_cohorts
from deepca.cohorts import CheckpointSplits, resolve_training_pairs

torch = test_cohorts.torch


class UnifiedEvaluationTests(unittest.TestCase):
    setUp = test_cohorts.UnifiedCohortTests.setUp
    tearDown = test_cohorts.UnifiedCohortTests.tearDown
    dataset_config = test_cohorts.UnifiedCohortTests.dataset_config
    write_projection = test_cohorts.UnifiedCohortTests.write_projection
    write_gt = test_cohorts.UnifiedCohortTests.write_gt
    make_cohort = test_cohorts.UnifiedCohortTests.make_cohort

    def test_reject_patient_leakage(self):
        manifest = {'expected_vessel': 'both', 'splits': {
            'train': {'case_ids': ['rca_0001']},
            'val': {'case_ids': []}, 'test': {'case_ids': ['lca_0001']}}}
        with self.assertRaisesRegex(ValueError, 'patient split leakage'):
            CheckpointSplits(manifest)

    @unittest.skipIf(torch is None, 'PyTorch unavailable')
    def test_unified_test_predictions_and_contract(self):
        import copy
        import numpy as np
        import yaml
        import evaluate_unified
        from deepca.modeling import build_generator, architecture_metadata

        config = self.make_cohort()
        _, manifest = resolve_training_pairs(config['data'])
        config['model'] = {'volume_size': 32, 'generator': {'base_filters': 8}}
        model = build_generator(config, 'cpu')
        checkpoint = self.root / 'checkpoint.pt'
        torch.save({'generator': model.state_dict(), 'config': copy.deepcopy(config),
                    'resolved_splits': manifest, 'architecture': architecture_metadata(model)},
                   checkpoint)
        output = self.root / 'evaluation'
        config['evaluation'] = {'checkpoint': str(checkpoint), 'device': 'cpu',
                                'output_dir': str(output), 'save_predictions': True,
                                'view_counts': [2], 'split': 'test'}
        path = self.root / 'evaluation.yaml'
        path.write_text(yaml.safe_dump(config))
        # Original splits and training files are not needed to evaluate held-out cases.
        for vessel, source in config['data']['sources'].items():
            Path(source['split_json']).unlink()
            for number in (1, 2):
                (Path(source['ground_truth_root']) / vessel / f'{number}.npz').unlink()
        self.assertEqual(evaluate_unified.main(['--config', str(path)]), 0)
        results = json.loads((output / 'test' / 'per_case.json').read_text())['results']
        self.assertEqual({row['case_id'] for row in results}, {'rca_0003', 'lca_0003'})
        for vessel in ('rca', 'lca'):
            prediction = output / 'test' / 'predictions' / vessel / 'views_2' / f'{vessel}_0003.npz'
            self.assertTrue(prediction.exists())
            with np.load(prediction, allow_pickle=False) as saved:
                self.assertEqual(str(saved['case_id']), f'{vessel}_0003')
        config['data']['sources']['lca']['calibration']['fallback_detector_pixel_spacing_mm'] = 0.7
        config['evaluation']['output_dir'] = str(self.root / 'mismatch')
        path.write_text(yaml.safe_dump(config))
        self.assertNotEqual(evaluate_unified.main(['--config', str(path)]), 0)
