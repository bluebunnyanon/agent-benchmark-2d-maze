import json
import unittest
from pathlib import Path


MAZE_JSON_DIR = Path(__file__).resolve().parents[1] / 'maze_jsons'


def _load_m1_specs():
	return [
		(path.name, json.loads(path.read_text(encoding='utf-8')))
		for path in sorted((MAZE_JSON_DIR / 'M1').glob('*.json'))
	]


def _parse_m1_name(file_name):
	stem = Path(file_name).stem
	parts = stem.split('_')
	if len(parts) < 4:
		raise ValueError(f'Unexpected M1 filename: {file_name}')
	size, structure_type, mechanism, variant = parts
	return size, structure_type, mechanism, variant


class TestM1MazeTypes(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.specs = _load_m1_specs()

	def test_expected_number_of_variants(self):
		"""Tests that M1 contains the expected number of maze variants."""
		self.assertEqual(len(self.specs), 10)

	def test_naming_scheme(self):
		"""Tests that each M1 filename matches its dimensions and kr mechanism type."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				size, structure_type, mechanism, variant = _parse_m1_name(file_name)
				width, height = spec['maze']['dimensions']

				self.assertEqual(size, f'{width}x{height}')
				self.assertIn(structure_type, {'corridor', 'dense'})
				self.assertEqual(mechanism, 'kr')
				self.assertIn(variant, {'0', '1'})

	def test_has_one_red_key_and_one_red_door(self):
		"""Tests that each M1 maze has exactly one red key and one red door."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				mechanisms = spec['mechanisms']
				self.assertEqual(mechanisms['switches'], [])
				self.assertEqual(mechanisms['gates'], [])
				self.assertEqual(len(mechanisms['keys']), 1)
				self.assertEqual(len(mechanisms['doors']), 1)

				self.assertEqual(
					mechanisms['keys'],
					[
						{
							'id': 'kR',
							'position': mechanisms['keys'][0]['position'],
							'color': 'red',
						}
					],
				)
				self.assertEqual(
					mechanisms['doors'],
					[
						{
							'id': 'DR',
							'position': mechanisms['doors'][0]['position'],
							'color': 'red',
							'requires_key': 'red',
							'initial_state': 'locked',
						}
					],
				)

	def test_has_no_extra_mechanism_groups(self):
		"""Tests that M1 mechanisms only use the standard mechanism groups."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				self.assertEqual(
					set(spec['mechanisms']),
					{'keys', 'doors', 'switches', 'gates'},
				)
