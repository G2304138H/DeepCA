from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from deepca.cohorts import resolve_training_pairs, training_sources
from deepca.data import ImageCASDataset
from tests import test_data

torch = test_data.torch


class UnifiedCohortTests(unittest.TestCase):
    setUp = test_data.ImageCASDataTestCase.setUp
    tearDown = test_data.ImageCASDataTestCase.tearDown
    dataset_config = test_data.ImageCASDataTestCase.dataset_config
    write_projection = test_data.ImageCASDataTestCase.write_projection
    write_gt = test_data.ImageCASDataTestCase.write_gt

    def make_cohort(self, conflict=False):
        config = self.dataset_config()
        data = config['data']
        data.update(vessel_type='both', sources={})
        for vessel, spacing in [('rca', 0.55), ('lca', 0.65)]:
            projections = self.root / 'projections' / vessel
            for number in (1, 2, 3):
                self.write_projection(projections / f'{vessel}_{number:04d}.npz',
                                      vessel=vessel, number=number)
                self.write_gt(self.root / 'gt' / vessel / f'{number}.npz')
            split = self.root / f'{vessel}.json'
            split.write_text(json.dumps({'train': [1], 'val': [2], 'test': [3]}
                if not conflict or vessel == 'rca' else
                {'train': [1], 'val': [3], 'test': [2]}))
            data['sources'][vessel] = {
                'projection_root': str(projections), 'ground_truth_root': str(self.root / 'gt'),
                'split_json': str(split),
                'calibration': {'fallback_detector_pixel_spacing_mm': spacing}}
        return config

    def test_unified_pairs_and_mixed_batch(self):
        config = self.make_cohort()
        pairs, manifest = resolve_training_pairs(config['data'])
        self.assertEqual([p.case_id for p in pairs['train']], ['rca_0001', 'lca_0001'])
        self.assertEqual(manifest['patient_split_moves'], [])
        if torch is None:
            self.skipTest('PyTorch is unavailable')
        dataset = ImageCASDataset(pairs['train'], config, training=True)
        from deepca.data import load_projection_npz
        with mock.patch('deepca.data.load_projection_npz', wraps=load_projection_npz) as loader:
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
        self.assertEqual(batch['vessel_type'], ['rca', 'lca'])
        self.assertEqual([call.kwargs['fallback_detector_pixel_spacing_mm']
                          for call in loader.call_args_list], [0.55, 0.65])

    def test_patient_conflicts_rejected_by_default(self):
        config = self.make_cohort(conflict=True)
        with self.assertRaisesRegex(ValueError, 'patient split leakage'):
            resolve_training_pairs(config['data'])

    def test_holdout_priority_never_leaks_patient(self):
        config = self.make_cohort(conflict=True)
        # Keep a validation patient after promoting both conflicting IDs to test.
        for source in config['data']['sources'].values():
            path = self.root / Path(source['split_json']).name
            splits = json.loads(path.read_text())
            splits['val'].append(4)
            path.write_text(json.dumps(splits))
            vessel = path.stem
            self.write_projection(Path(source['projection_root']) / f'{vessel}_0004.npz',
                                  vessel=vessel, number=4)
            self.write_gt(self.root / 'gt' / vessel / '4.npz')
        config['data']['patient_split_policy'] = 'holdout_priority'
        pairs, manifest = resolve_training_pairs(config['data'])
        self.assertEqual(len(manifest['patient_split_moves']), 2)
        self.assertEqual({p.case_number for p in pairs['test']}, {2, 3})
        seen = {}
        for split, items in pairs.items():
            for pair in items:
                self.assertEqual(seen.setdefault(pair.case_number, split), split)

    def test_requires_both_sources(self):
        with self.assertRaisesRegex(ValueError, 'exactly rca and lca'):
            training_sources({'vessel_type': 'both', 'sources': {'rca': {}}})

    @unittest.skipIf(torch is None, 'PyTorch is unavailable')
    def test_trainer_dry_run_training_and_resume(self):
        import train
        import yaml
        config = self.make_cohort()
        config['model'] = {'volume_size': 32,
                           'generator': {'base_filters': 8},
                           'critic': {'channels': 2, 'dim': 32}}
        config['training'] = {'device': 'cpu', 'epochs': 1, 'workers': 0,
                              'batch_size': 2, 'drop_last': False,
                              'mixed_precision': False, 'critic_steps_per_generator': 1}
        path = self.root / 'config.yaml'
        path.write_text(yaml.safe_dump(config))
        output = self.root / 'out'
        with mock.patch('train.build_models', side_effect=AssertionError('dry run built model')):
            self.assertEqual(train.main(['--config', str(path), '--dry-run']), 0)
        self.assertTrue((output / 'resolved_cases.json').exists())
        self.assertEqual(train.main(['--config', str(path)]), 0)
        record = json.loads((output / 'epochs.jsonl').read_text().splitlines()[0])
        self.assertIn('rca_l1_loss', record['validation'])
        self.assertIn('lca_l1_loss', record['validation'])
        self.assertTrue((output / 'checkpoints' / 'best.pt').exists())
        config['training']['epochs'] = 2
        path.write_text(yaml.safe_dump(config))
        self.assertEqual(train.main(['--config', str(path), '--resume',
                                    str(output / 'checkpoints' / 'last.pt')]), 0)
        records = (output / 'epochs.jsonl').read_text().splitlines()
        self.assertEqual(len(records), 2)
        self.assertEqual(json.loads(records[-1])['epoch'], 1)
