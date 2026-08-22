import json
import unittest
from collections import deque
from pathlib import Path


MAZE_JSON_DIR = Path(__file__).resolve().parents[1] / 'maze_jsons'

MECHANISM_CHAIN = {
	'kr': ('kR', 'DR'),
	'sg': ('s1', 'g1'),
	'kb': ('kB', 'DB'),
	'sb': ('kB', 'DB'),
}


def _load_m6_specs():
	specs = []
	for path in sorted((MAZE_JSON_DIR / 'M6').glob('*.json')):
		specs.append((path.name, json.loads(path.read_text(encoding='utf-8'))))
	return specs


def _parse_m6_name(file_name):
	stem = Path(file_name).stem
	parts = stem.split('_')
	if len(parts) < 4:
		raise ValueError(f'Unexpected M6 filename: {file_name}')
	size, structure_type = parts[:2]
	variant = parts[-1]
	mechanisms = parts[2:-1]
	return size, structure_type, mechanisms, variant


def _mechanism_positions(spec):
	positions = {}
	for group_name in ('keys', 'doors', 'switches', 'gates'):
		for mechanism in spec['mechanisms'][group_name]:
			positions[mechanism['id']] = tuple(mechanism['position'])
	return positions


def _walkable_cells(spec):
	width, height = spec['maze']['dimensions']
	walls = {tuple(wall) for wall in spec['maze']['walls']}
	mechanism_cells = set(_mechanism_positions(spec).values())
	# Mechanisms may replace wall cells, e.g. locked doors or closed gates.
	walls -= mechanism_cells

	walkable = set()
	for y in range(1, height - 1):
		for x in range(1, width - 1):
			if (x, y) not in walls:
				walkable.add((x, y))
	return walkable


def _find_path(spec, start, goal):
	walkable = _walkable_cells(spec)
	walkable.add(start)
	walkable.add(goal)

	queue = deque([start])
	previous = {start: None}
	while queue:
		x, y = queue.popleft()
		if (x, y) == goal:
			break
		for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
			if neighbor in walkable and neighbor not in previous:
				previous[neighbor] = (x, y)
				queue.append(neighbor)

	if goal not in previous:
		return None

	path = []
	current = goal
	while current is not None:
		path.append(current)
		current = previous[current]
	return list(reversed(path))


def _chain_for_file(file_name):
	_, _, mechanisms, _ = _parse_m6_name(file_name)
	chain = []
	for mechanism in mechanisms:
		chain.extend(MECHANISM_CHAIN[mechanism])
	return chain


class TestM6MazeTypes(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.specs = _load_m6_specs()

	def test_expected_number_of_variants(self):
		"""Tests that M6 contains the expected number of maze variants."""
		self.assertEqual(len(self.specs), 10)

	def test_naming_scheme(self):
		"""Tests that each M6 filename matches its dimensions and kr-sg-kb chain."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				size, structure_type, mechanisms, variant = _parse_m6_name(file_name)
				width, height = spec['maze']['dimensions']

				self.assertEqual(size, f'{width}x{height}')
				self.assertIn(structure_type, {'corridor', 'dense'})
				self.assertEqual(mechanisms, ['kr', 'sg', 'kb'])
				self.assertIn(variant, {'0', '1'})

	def test_path_to_goal_exists(self):
		"""Tests that each M6 maze has at least one walkable path to the goal."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				start = tuple(spec['maze']['start'])
				goal = tuple(spec['maze']['goal'])

				self.assertIsNotNone(
					_find_path(spec, start, goal),
					'expected at least one walkable path from start to goal',
				)

	def test_mechanism_chain_order_on_path_to_goal(self):
		"""Tests that M6 chain waypoints are reachable in filename order."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				positions = _mechanism_positions(spec)
				chain = _chain_for_file(file_name)
				waypoints = [positions[mechanism_id] for mechanism_id in chain]
				points = [tuple(spec['maze']['start']), *waypoints, tuple(spec['maze']['goal'])]

				for start, goal in zip(points, points[1:]):
					path = _find_path(spec, start, goal)
					self.assertIsNotNone(
						path,
						f'expected ordered path segment from {start} to {goal}',
					)

	def test_mechanism_pairs_follow_filename_order(self):
		"""Tests that M6 mechanism pairs expand to adjacent ordered waypoints."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				positions = _mechanism_positions(spec)
				chain = _chain_for_file(file_name)
				for earlier, later in zip(chain, chain[1:]):
					self.assertNotEqual(
						positions[earlier],
						positions[later],
						f'{earlier} and {later} should not share a maze cell',
					)

				for mechanism, opener, follower in (
					('kr', 'kR', 'DR'),
					('sg', 's1', 'g1'),
					('kb', 'kB', 'DB'),
				):
					if opener not in chain:
						continue
					opener_index = chain.index(opener)
					follower_index = chain.index(follower)
					self.assertEqual(
						follower_index,
						opener_index + 1,
						f'{mechanism} should expand to adjacent opener/follower waypoints',
					)
					for later in chain[follower_index + 1:]:
						self.assertLess(
							follower_index,
							chain.index(later),
							f'{follower} should appear before {later} in the filename-derived chain',
						)
