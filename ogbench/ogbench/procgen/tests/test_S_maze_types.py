import json
import unittest
from pathlib import Path

from BFS_solver import find_all_paths


MAZE_JSON_DIR = Path(__file__).resolve().parents[1] / 'maze_jsons'
MECHANISM_KEYS = ('keys', 'doors', 'switches', 'gates')


def _load_maze_specs(maze_type):
	return [
		json.loads(path.read_text(encoding='utf-8'))
		for path in sorted((MAZE_JSON_DIR / maze_type).glob('*.json'))
	]


def _assert_common_navigation_maze(test_case, spec):
	maze = spec['maze']
	width, height = maze['dimensions']
	start = maze['start']
	goal = maze['goal']
	walls = {tuple(wall) for wall in maze['walls']}

	test_case.assertEqual(spec['goal']['type'], 'reach_position')
	test_case.assertEqual(len(start), 2)
	test_case.assertEqual(len(goal), 2)
	test_case.assertNotEqual(start, goal)
	for label, point in (('start', start), ('goal', goal)):
		x, y = point
		test_case.assertGreaterEqual(x, 0, label)
		test_case.assertLess(x, width, label)
		test_case.assertGreaterEqual(y, 0, label)
		test_case.assertLess(y, height, label)
		test_case.assertNotIn(tuple(point), walls, label)


def _assert_no_mechanisms(test_case, spec):
	mechanisms = spec['mechanisms']
	for key in MECHANISM_KEYS:
		test_case.assertEqual(mechanisms[key], [], key)
	for key, value in mechanisms.items():
		test_case.assertEqual(value, [], key)
	test_case.assertEqual(spec['rules']['hidden_mechanisms'], [])
	test_case.assertEqual(spec['metadata']['chain_pattern'], 'none')


class SimpleMazeAssertions:
	maze_type = None
	expected_dimensions = None
	expected_difficulty_tier = None
	expected_wall_topology = None

	@classmethod
	def setUpClass(cls):
		cls.specs = _load_maze_specs(cls.maze_type)

	def test_expected_number_of_variants(self):
		"""Tests that each simple maze type has exactly two JSON variants."""
		self.assertEqual(len(self.specs), 2)

	def test_navigation_contract(self):
		"""Tests that each simple maze has valid start and goal navigation fields."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				_assert_common_navigation_maze(self, spec)

	def test_has_no_mechanisms(self):
		"""Tests that simple mazes do not define keys, doors, switches, or gates."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				_assert_no_mechanisms(self, spec)

	def test_matches_maze_type_metadata(self):
		"""Tests that each simple maze matches its expected dimensions and metadata."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertEqual(spec['maze']['dimensions'], self.expected_dimensions)
				self.assertEqual(spec['difficulty_tier'], self.expected_difficulty_tier)
				self.assertEqual(spec['metadata']['wall_topology'], self.expected_wall_topology)

	@unittest.skip(':TODO add BFS solver uniqueness check')
	def test_bfs_solver_finds_one_and_only_one_path_to_goal(self):
		"""TODO: verify each simple maze has exactly one path from start to goal."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertEqual(len(find_all_paths(spec)), 1)


class TestS1DSMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S1'
	expected_dimensions = [8, 8]
	expected_difficulty_tier = 1
	expected_wall_topology = 'open'

	def test_has_no_interior_walls(self):
		"""Tests that S1 open-room mazes have no interior wall cells."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertEqual(spec['maze']['walls'], [])


class TestS2SmallCorridorMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S2'
	expected_dimensions = [8, 8]
	expected_difficulty_tier = 2
	expected_wall_topology = 'winding'

	def test_has_corridor_walls(self):
		"""Tests that S2 small corridor mazes define interior wall cells."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertGreater(len(spec['maze']['walls']), 0)


class TestS3MediumCorridorMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S3'
	expected_dimensions = [10, 10]
	expected_difficulty_tier = 3
	expected_wall_topology = 'winding'

	def test_has_corridor_walls(self):
		"""Tests that S3 medium corridor mazes define interior wall cells."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertGreater(len(spec['maze']['walls']), 0)


class TestS4MediumDenseMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S4'
	expected_dimensions = [10, 10]
	expected_difficulty_tier = 4
	expected_wall_topology = 'dense_dead_ends'

	def test_has_dense_walls(self):
		"""Tests that S4 medium dense mazes have the expected dense wall count."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertGreaterEqual(len(spec['maze']['walls']), 30)


class TestS5LargeCorridorMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S5'
	expected_dimensions = [14, 14]
	expected_difficulty_tier = 5
	expected_wall_topology = 'winding'

	def test_has_corridor_walls(self):
		"""Tests that S5 large corridor mazes define interior wall cells."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertGreater(len(spec['maze']['walls']), 0)


class TestS6LargeDenseMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S6'
	expected_dimensions = [14, 14]
	expected_difficulty_tier = 6
	expected_wall_topology = 'dense_dead_ends'

	def test_has_dense_walls(self):
		"""Tests that S6 large dense mazes have the expected dense wall count."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertGreaterEqual(len(spec['maze']['walls']), 60)
