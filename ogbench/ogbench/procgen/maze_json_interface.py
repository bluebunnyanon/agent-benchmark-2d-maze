"""JSON interface for procedural maze specifications.

This module parses maze-task JSON files (like the validation schema shared by
the user) and generates a validated 2D maze representation.

Coordinate convention:
- JSON uses ``[x, y]``
- Internal grids use ``grid[y, x]``

Grid convention:
- ``0``: free cell
- ``1``: wall cell
"""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ogbench.locomaze.maze import make_maze_env


FREE = 0
WALL = 1


@dataclass(frozen=True)
class ParsedMaze:
	"""Parsed and validated maze specification."""

	task_id: str
	version: str
	seed: int | None
	width: int
	height: int
	start: tuple[int, int]
	goal: tuple[int, int]
	wall_cells: tuple[tuple[int, int], ...]
	mechanisms: Mapping[str, Any]
	rules: Mapping[str, Any]
	goal_spec: Mapping[str, Any]
	metadata: Mapping[str, Any]
	max_steps: int | None

	def to_wall_grid(self, add_boundary_walls: bool = True) -> np.ndarray:
		"""Return a ``(height, width)`` grid with ``0=free``, ``1=wall``."""
		grid = np.zeros((self.height, self.width), dtype=np.int8)

		if add_boundary_walls:
			grid[0, :] = WALL
			grid[-1, :] = WALL
			grid[:, 0] = WALL
			grid[:, -1] = WALL

		for x, y in self.wall_cells:
			grid[y, x] = WALL

		return grid

	def to_symbol_grid(
		self,
		add_boundary_walls: bool = True,
		include_mechanisms: bool = False,
	) -> list[list[str]]:
		"""Return a printable symbol grid.

		Symbols:
		- ``#`` wall
		- ``.`` free
		- ``S`` start
		- ``G`` goal
		- ``K`` key (optional)
		- ``D`` door (optional)
		"""
		grid = self.to_wall_grid(add_boundary_walls=add_boundary_walls)
		out = [['#' if grid[y, x] == WALL else '.' for x in range(self.width)] for y in range(self.height)]

		if include_mechanisms:
			for key_obj in self.mechanisms.get('keys', []):
				if isinstance(key_obj, Mapping) and 'position' in key_obj:
					x, y = _parse_xy(key_obj['position'], field='mechanisms.keys[].position')
					if 0 <= x < self.width and 0 <= y < self.height and out[y][x] == '.':
						out[y][x] = 'K'

			for door_obj in self.mechanisms.get('doors', []):
				if isinstance(door_obj, Mapping) and 'position' in door_obj:
					x, y = _parse_xy(door_obj['position'], field='mechanisms.doors[].position')
					if 0 <= x < self.width and 0 <= y < self.height and out[y][x] == '.':
						out[y][x] = 'D'

		sx, sy = self.start
		gx, gy = self.goal
		out[sy][sx] = 'S'
		out[gy][gx] = 'G'
		return out

	def to_ascii(self, add_boundary_walls: bool = True, include_mechanisms: bool = False) -> str:
		"""Render the maze as ASCII text."""
		sym = self.to_symbol_grid(
			add_boundary_walls=add_boundary_walls,
			include_mechanisms=include_mechanisms,
		)
		return '\n'.join(''.join(row) for row in sym)


def parse_maze_json(payload: Mapping[str, Any]) -> ParsedMaze:
	"""Parse a maze JSON payload into a validated :class:`ParsedMaze`."""
	maze = _expect_mapping(payload.get('maze'), 'maze')

	width, height = _parse_dimensions(maze.get('dimensions'))
	start = _parse_xy(maze.get('start'), field='maze.start')
	goal = _parse_xy(maze.get('goal'), field='maze.goal')
	wall_cells = _parse_wall_list(maze.get('walls', []))

	_validate_in_bounds(start, width, height, 'maze.start')
	_validate_in_bounds(goal, width, height, 'maze.goal')
	for idx, wall in enumerate(wall_cells):
		_validate_in_bounds(wall, width, height, f'maze.walls[{idx}]')

	tmp_grid = np.zeros((height, width), dtype=np.int8)
	tmp_grid[0, :] = WALL
	tmp_grid[-1, :] = WALL
	tmp_grid[:, 0] = WALL
	tmp_grid[:, -1] = WALL
	for x, y in wall_cells:
		tmp_grid[y, x] = WALL

	if tmp_grid[start[1], start[0]] == WALL:
		raise ValueError('maze.start cannot be placed on a wall cell (including implicit boundaries).')
	if tmp_grid[goal[1], goal[0]] == WALL:
		raise ValueError('maze.goal cannot be placed on a wall cell (including implicit boundaries).')

	task_id = str(payload.get('task_id', ''))
	version = str(payload.get('version', ''))
	seed = _optional_int(payload.get('seed'), 'seed')
	max_steps = _optional_int(payload.get('max_steps'), 'max_steps')

	mechanisms = _expect_mapping(payload.get('mechanisms', {}), 'mechanisms')
	rules = _expect_mapping(payload.get('rules', {}), 'rules')
	goal_spec = _expect_mapping(payload.get('goal', {}), 'goal')
	metadata = _expect_mapping(payload.get('metadata', {}), 'metadata')

	parsed = ParsedMaze(
		task_id=task_id,
		version=version,
		seed=seed,
		width=width,
		height=height,
		start=start,
		goal=goal,
		wall_cells=tuple(wall_cells),
		mechanisms=mechanisms,
		rules=rules,
		goal_spec=goal_spec,
		metadata=metadata,
		max_steps=max_steps,
	)

	_validate_mechanism_positions(parsed)

	return parsed


def parse_maze_json_file(path: str | Path) -> ParsedMaze:
	"""Load and parse a maze JSON file."""
	file_path = Path(path)
	with file_path.open('r', encoding='utf-8') as f:
		payload = json.load(f)
	return parse_maze_json(payload)


def parse_maze_json_string(raw_json: str) -> ParsedMaze:
	"""Parse a raw JSON string containing a maze specification."""
	payload = json.loads(raw_json)
	return parse_maze_json(payload)


def generate_maze_grid(payload: Mapping[str, Any], add_boundary_walls: bool = True) -> np.ndarray:
	"""Convenience wrapper: parse payload and directly return wall grid."""
	return parse_maze_json(payload).to_wall_grid(add_boundary_walls=add_boundary_walls)


def make_pointymaze_env_from_json(
	path: str | Path,
	*,
	json_origin: str = 'top_left',
	maze_unit: float = 1.4,
	maze_height: float = 0.5,
	render_mode: str = 'rgb_array',
	width: int = 512,
	height: int = 512,
	ob_type: str = 'states',
	add_noise_to_goal: bool = False,
	terminate_at_goal: bool = True,
):
	"""Create a custom pointy maze env from a JSON maze file.

	The JSON ``maze.start`` and ``maze.goal`` are used as the default task.

	Args:
		path: Path to maze JSON.
		json_origin: Coordinate convention used by JSON positions. One of
			``'top_left'`` or ``'bottom_left'``.
	"""
	if json_origin not in {'top_left', 'bottom_left'}:
		raise ValueError("json_origin must be either 'top_left' or 'bottom_left'.")

	parsed = parse_maze_json_file(path)
	maze_map = parsed.to_wall_grid(add_boundary_walls=True)
	mechanisms = _convert_mechanisms_for_origin(parsed.mechanisms, parsed.height, json_origin)
	if json_origin == 'top_left':
		# Locomaze treats row 0 as bottom; most grid JSONs use row 0 as top.
		maze_map = np.flipud(maze_map)

	# Convert [x, y] -> (i, j) = (row, col).
	if json_origin == 'top_left':
		start_ij = (parsed.height - 1 - parsed.start[1], parsed.start[0])
		goal_ij = (parsed.height - 1 - parsed.goal[1], parsed.goal[0])
	else:
		start_ij = (parsed.start[1], parsed.start[0])
		goal_ij = (parsed.goal[1], parsed.goal[0])

	custom_tasks = [[start_ij, goal_ij]]

	env = make_maze_env(
		'pointy',
		'maze',
		maze_type='custom',
		maze_map=maze_map,
		custom_tasks=custom_tasks,
		mechanisms=mechanisms,
		mechanism_rules=parsed.rules,
		maze_unit=maze_unit,
		maze_height=maze_height,
		ob_type=ob_type,
		add_noise_to_goal=add_noise_to_goal,
		terminate_at_goal=terminate_at_goal,
		render_mode=render_mode,
		width=width,
		height=height,
	)
	return env, parsed


def _convert_mechanisms_for_origin(
	mechanisms: Mapping[str, Any],
	height: int,
	json_origin: str,
) -> Mapping[str, Any]:
	"""Convert mechanism coordinates from JSON origin to locomaze origin."""
	if json_origin == 'bottom_left':
		return mechanisms

	converted = copy.deepcopy(dict(mechanisms))

	for key in ('keys', 'doors', 'switches', 'gates', 'blocks', 'teleporters', 'hazards'):
		items = converted.get(key)
		if not isinstance(items, list):
			continue
		for item in items:
			if not isinstance(item, Mapping):
				continue
			for pos_key in ('position', 'pos', 'start', 'goal', 'target', 'location', 'from', 'to', 'src', 'dst'):
				value = item.get(pos_key)
				if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
					x = int(value[0])
					y = int(value[1])
					item[pos_key] = [x, height - 1 - y]

	return converted


def _expect_mapping(value: Any, field: str) -> Mapping[str, Any]:
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		raise TypeError(f'{field} must be a JSON object/dict, got {type(value).__name__}.')
	return value


def _parse_dimensions(value: Any) -> tuple[int, int]:
	if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
		raise ValueError('maze.dimensions must be [width, height].')
	width = _parse_int(value[0], 'maze.dimensions[0]')
	height = _parse_int(value[1], 'maze.dimensions[1]')
	if width < 2 or height < 2:
		raise ValueError('maze.dimensions must each be >= 2.')
	return width, height


def _parse_xy(value: Any, field: str) -> tuple[int, int]:
	if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
		raise ValueError(f'{field} must be [x, y].')
	x = _parse_int(value[0], f'{field}[0]')
	y = _parse_int(value[1], f'{field}[1]')
	return x, y


def _parse_wall_list(value: Any) -> list[tuple[int, int]]:
	if value is None:
		return []
	if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
		raise TypeError('maze.walls must be a list of [x, y] coordinates.')

	walls: list[tuple[int, int]] = []
	seen: set[tuple[int, int]] = set()
	for idx, item in enumerate(value):
		xy = _parse_xy(item, field=f'maze.walls[{idx}]')
		if xy not in seen:
			walls.append(xy)
			seen.add(xy)
	return walls


def _parse_int(value: Any, field: str) -> int:
	if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
		raise TypeError(f'{field} must be an integer.')
	return int(value)


def _optional_int(value: Any, field: str) -> int | None:
	if value is None:
		return None
	return _parse_int(value, field)


def _validate_in_bounds(xy: tuple[int, int], width: int, height: int, field: str) -> None:
	x, y = xy
	if x < 0 or y < 0 or x >= width or y >= height:
		raise ValueError(f'{field}={xy} is out of bounds for dimensions ({width}, {height}).')


def _validate_mechanism_positions(parsed: ParsedMaze) -> None:
	"""Best-effort validation for mechanism positions found in the JSON.

	This walks common coordinate keys recursively and verifies all discovered
	positions are in bounds and not on hard walls.
	"""
	wall_grid = parsed.to_wall_grid(add_boundary_walls=True)
	for mech_name, mech_value in parsed.mechanisms.items():
		for idx, xy in enumerate(_extract_positions(mech_value)):
			_validate_in_bounds(xy, parsed.width, parsed.height, f'mechanisms.{mech_name}[{idx}]')
			if wall_grid[xy[1], xy[0]] == WALL:
				raise ValueError(f'mechanisms.{mech_name}[{idx}]={xy} is placed on a wall cell.')


def _extract_positions(node: Any) -> list[tuple[int, int]]:
	"""Recursively collect 2D positions from flexible mechanism schemas."""
	positions: list[tuple[int, int]] = []
	if isinstance(node, Mapping):
		for key, value in node.items():
			key_l = str(key).lower()
			if key_l in {
				'position',
				'pos',
				'start',
				'goal',
				'target',
				'location',
				'from',
				'to',
				'src',
				'dst',
				'source',
				'destination',
			}:
				try:
					positions.append(_parse_xy(value, field=f'mechanism.{key}'))
					continue
				except Exception:
					# Fall through to recursive descent for nested structures.
					pass
			positions.extend(_extract_positions(value))
	elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
		for item in node:
			positions.extend(_extract_positions(item))
	return positions

