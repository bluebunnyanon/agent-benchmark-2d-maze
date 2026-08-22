import tempfile
import xml.etree.ElementTree as ET
import re

import mujoco
import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple

from ogbench.locomaze.ant import AntEnv
from ogbench.locomaze.humanoid import HumanoidEnv
from ogbench.locomaze.point import PointEnv
from ogbench.locomaze.pointy import PointyEnv


def make_maze_env(loco_env_type, maze_env_type, *args, **kwargs):
    """Factory function for creating a maze environment.

    Args:
        loco_env_type: Locomotion environment type. One of 'point', 'ant', or 'humanoid'.
        maze_env_type: Maze environment type. Either 'maze' or 'ball'.
        *args: Additional arguments to pass to the target class.
        **kwargs: Additional keyword arguments to pass to the target class.
    """
    if loco_env_type == 'point':
        loco_env_class = PointEnv
    elif loco_env_type == 'pointy':
        loco_env_class = PointyEnv
    elif loco_env_type == 'ant':
        loco_env_class = AntEnv
    elif loco_env_type == 'humanoid':
        loco_env_class = HumanoidEnv
    else:
        raise ValueError(f'Unknown locomotion environment type: {loco_env_type}')

    class MazeEnv(loco_env_class):
        """Maze environment.

        It inherits from the locomotion environment and adds a maze to it.
        """

        def __init__(
            self,
            maze_type='large',
            maze_map=None,
            custom_tasks=None,
            mechanisms=None,
            mechanism_rules=None,
            maze_unit=4.0,
            maze_height=0.5,
            terminate_at_goal=True,
            success_timing='post',
            ob_type='states',
            add_noise_to_goal=True,
            reward_task_id=None,
            use_oracle_rep=False,
            *args,
            **kwargs,
        ):
            """Initialize the maze environment.

            Args:
                maze_type: Maze type. One of 'arena', 'medium', 'large', 'giant', 'teleport',
                    'windingcorridor', 'multipath', or 'custom'.
                maze_map: Optional custom maze map with shape (H, W), where 0 is free and 1 is wall.
                    If provided, this overrides built-in maze layouts.
                custom_tasks: Optional list of tasks for custom mazes in the format
                    [[(init_i, init_j), (goal_i, goal_j)], ...]. If omitted for a custom maze,
                    one default task is created from the first and last free cells.
                maze_unit: Size of a maze unit block.
                maze_height: Height of the maze walls.
                terminate_at_goal: Whether to terminate the episode when the goal is reached.
                success_timing: When to compute the success and terminated flags and the reward. Either 'pre' (before
                    taking the action) or 'post' (after taking the action).
                ob_type: Observation type. Either 'states' or 'pixels'.
                add_noise_to_goal: Whether to add noise to the goal position.
                reward_task_id: Task ID for single-task RL. If this is not None, the environment operates in a
                    single-task mode with the specified task ID. The task ID must be either a valid task ID or 0, where
                    0 means using the default task.
                use_oracle_rep: Whether to use oracle goal representations.
                *args: Additional arguments to pass to the parent locomotion environment.
                **kwargs: Additional keyword arguments to pass to the parent locomotion environment.
            """
            self._maze_type = maze_type
            self._custom_tasks = custom_tasks
            self._raw_mechanisms = mechanisms if mechanisms is not None else {}
            self._mechanism_rules = mechanism_rules if mechanism_rules is not None else {}
            self._maze_unit = maze_unit
            self._maze_height = maze_height
            self._terminate_at_goal = terminate_at_goal
            self._success_timing = success_timing
            self._ob_type = ob_type
            self._add_noise_to_goal = add_noise_to_goal
            self._reward_task_id = reward_task_id
            self._use_oracle_rep = use_oracle_rep

            assert ob_type in ['states', 'pixels']
            assert success_timing in ['pre', 'post']

            # Define constants.
            self._offset_x = 4
            self._offset_y = 4
            self._noise = 1
            self._goal_tol = 1.0 if loco_env_type == 'point' else 0.5
            self._key_pickup_radius = 0.40 * self._maze_unit
            self._door_open_radius = 0.55 * self._maze_unit
            self._switch_toggle_radius = 0.45 * self._maze_unit
            # Effective collision disk for point/pointy movement checks.
            # Keep this smaller than the visual/body geom to avoid over-blocking.
            self._point_collision_radius = 0.18 * self._maze_unit

            # Key/door mechanism state.
            self._key_items = []
            self._door_items = []
            self._inventory_key_colors = set()

            # Define maze map.
            self._teleport_info = None
            custom_maze_map = maze_map
            maze_map_data = None
            if custom_maze_map is not None:
                maze_map_data = np.asarray(custom_maze_map)
                if maze_map_data.ndim != 2:
                    raise ValueError(f'Custom maze_map must be 2D, got shape {maze_map_data.shape}.')
                if not np.all((maze_map_data == 0) | (maze_map_data == 1)):
                    raise ValueError('Custom maze_map must contain only 0 (free) and 1 (wall).')
                if self._maze_type != 'custom':
                    # A custom map was explicitly provided; mark as custom layout for task handling.
                    self._maze_type = 'custom'
            elif self._maze_type == 'arena':
                maze_map_data = [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type == 'medium':
                maze_map_data = [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type == 'large':
                maze_map_data = [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type == 'giant':
                maze_map_data = [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                    [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type == 'teleport':
                maze_map_data = [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                    [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
                    [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
                self._teleport_info = dict(
                    teleport_in_ijs=[(4, 6), (5, 1)],
                    teleport_out_ijs=[(1, 7), (6, 1), (6, 10)],
                    teleport_radius=1,
                )
                self._teleport_info['teleport_in_xys'] = [
                    self.ij_to_xy(ij) for ij in self._teleport_info['teleport_in_ijs']
                ]
                self._teleport_info['teleport_out_xys'] = [
                    self.ij_to_xy(ij) for ij in self._teleport_info['teleport_out_ijs']
                ]
            elif self._maze_type == 'multipath':
                # Three winding routes (top/middle/bottom) connect the same start/goal regions.
                maze_map_data = [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                    [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            elif self._maze_type in ('windingcorridor', 'longpath'):
                # A single winding corridor (serpentine path) with no branches.
                maze_map_data = [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self.maze_map = np.array(maze_map_data, dtype=np.int8)
            self._initialize_key_door_mechanisms()
            self._has_open_action = loco_env_type in ('point', 'pointy') and (
                len(self._door_items) > 0 or len(self._switch_items) > 0
            )

            # Update XML file.
            xml_file = self.xml_file
            tree = ET.parse(xml_file)
            self.update_tree(tree)
            _, maze_xml_file = tempfile.mkstemp(text=True, suffix='.xml')
            tree.write(maze_xml_file)

            super().__init__(xml_file=maze_xml_file, *args, **kwargs)
            self._bind_key_door_geoms()
            self._move_action_space = self.action_space

            if self._has_open_action:
                self.action_space = Tuple(
                    (
                        self._move_action_space,
                        Discrete(3),  # 0=no-op, 1=pickup, 2=open/use
                    )
                )

            # Make custom camera.
            if self.camera_id is None and self.camera_name is None:
                # Use a custom default view.
                camera = mujoco.MjvCamera()
                maze_h, maze_w = self.maze_map.shape
                camera.lookat[0] = 0.5 * self._maze_unit * (maze_w - 1) - self._offset_x
                camera.lookat[1] = 0.5 * self._maze_unit * (maze_h - 1) - self._offset_y
                camera.distance = 1.25 * self._maze_unit * max(maze_w - 2, maze_h - 2)
                camera.elevation = -90
                self.custom_camera = camera
            else:
                self.custom_camera = self.camera_id or self.camera_name

            # Set task goals.
            self.task_infos = []
            self.cur_task_id = None
            self.cur_task_info = None
            self.set_tasks()
            self.num_tasks = len(self.task_infos)
            self.cur_goal_xy = np.zeros(2)

            self.custom_renderer = None
            if self._ob_type == 'pixels':
                self.observation_space = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

                # Manually color the floor to enable the agent to infer its position from the observation.
                tex_grid = self.model.tex('grid')
                tex_height = tex_grid.height[0]
                tex_width = tex_grid.width[0]
                # MuJoCo 3.2.1 changed the attribute name from 'tex_rgb' to 'tex_data'.
                attr_name = 'tex_rgb' if hasattr(self.model, 'tex_rgb') else 'tex_data'
                tex_rgb = getattr(self.model, attr_name)[tex_grid.adr[0] : tex_grid.adr[0] + 3 * tex_height * tex_width]
                tex_rgb = tex_rgb.reshape(tex_height, tex_width, 3)
                for x in range(tex_height):
                    for y in range(tex_width):
                        min_value = 0
                        max_value = 192
                        r = int(x / tex_height * (max_value - min_value) + min_value)
                        g = int(y / tex_width * (max_value - min_value) + min_value)
                        tex_rgb[x, y, :] = [r, g, 128]
                self.initialize_renderer()
            else:
                ex_ob = self.get_ob()
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

        def update_tree(self, tree):
            """Update the XML tree to include the maze."""
            worldbody = tree.find('.//worldbody')
            asset = tree.find('.//asset')

            # Style pointy agent and goal marker for easier visibility.
            agent_marker_geom = tree.find('.//geom[@name="agent_marker_geom"]')
            if agent_marker_geom is not None:
                agent_marker_geom.set('rgba', '.92 .18 .18 1')
            target_material = tree.find('.//material[@name="target"]')
            if target_material is not None:
                target_material.set('rgba', '.20 .82 .28 1')

            # Add walls.
            for i in range(self.maze_map.shape[0]):
                for j in range(self.maze_map.shape[1]):
                    struct = self.maze_map[i, j]
                    if struct == 1:
                        ET.SubElement(
                            worldbody,
                            'geom',
                            name=f'block_{i}_{j}',
                            pos=f'{j * self._maze_unit - self._offset_x} {i * self._maze_unit - self._offset_y} {self._maze_height / 2 * self._maze_unit}',
                            size=f'{self._maze_unit / 2} {self._maze_unit / 2} {self._maze_height / 2 * self._maze_unit}',
                            type='box',
                            contype='1',
                            conaffinity='1',
                            material='wall',
                        )

            # Add key/door/switch/gate mechanism visuals.
            key_colors = {item['color'] for item in self._key_items}
            door_colors = {item['requires_key_color'] for item in self._door_items}
            for color in sorted(key_colors):
                key_material = f'key_{color}_material'
                if asset.find(f".//material[@name='{key_material}']") is None:
                    ET.SubElement(asset, 'material', name=key_material, rgba=self._rgba_for_color(color, alpha=1.0))
            for color in sorted(door_colors):
                door_material = f'door_{color}_material'
                if asset.find(f".//material[@name='{door_material}']") is None:
                    ET.SubElement(asset, 'material', name=door_material, rgba=self._rgba_for_color(color, alpha=1.0))
            switch_colors = {item['color'] for item in self._switch_items}
            for color in sorted(switch_colors):
                switch_material = f'switch_{color}_material'
                if asset.find(f".//material[@name='{switch_material}']") is None:
                    ET.SubElement(asset, 'material', name=switch_material, rgba=self._rgba_for_color(color, alpha=1.0))
            if len(self._gate_items) > 0 and asset.find(".//material[@name='gate_black_material']") is None:
                ET.SubElement(asset, 'material', name='gate_black_material', rgba=self._rgba_for_color('black', alpha=1.0))

            for item in self._key_items:
                x, y = item['xy']
                ring_name, stem_name, tooth_name = item['geom_names']
                z = 0.12 * self._maze_unit
                ET.SubElement(
                    worldbody,
                    'geom',
                    name=ring_name,
                    type='sphere',
                    size=f'{0.10 * self._maze_unit}',
                    pos=f'{x - 0.11 * self._maze_unit} {y} {z}',
                    material=f"key_{item['color']}_material",
                    contype='0',
                    conaffinity='0',
                )
                ET.SubElement(
                    worldbody,
                    'geom',
                    name=stem_name,
                    type='box',
                    size=f'{0.15 * self._maze_unit} {0.028 * self._maze_unit} {0.028 * self._maze_unit}',
                    pos=f'{x + 0.06 * self._maze_unit} {y} {z}',
                    material=f"key_{item['color']}_material",
                    contype='0',
                    conaffinity='0',
                )
                ET.SubElement(
                    worldbody,
                    'geom',
                    name=tooth_name,
                    type='box',
                    size=f'{0.03 * self._maze_unit} {0.03 * self._maze_unit} {0.028 * self._maze_unit}',
                    pos=f'{x + 0.21 * self._maze_unit} {y - 0.055 * self._maze_unit} {z}',
                    material=f"key_{item['color']}_material",
                    contype='0',
                    conaffinity='0',
                )

            for item in self._door_items:
                x, y = item['xy']
                sx, sy = item['door_size_xy']
                ET.SubElement(
                    worldbody,
                    'geom',
                    name=item['geom_name'],
                    type='box',
                    size=f'{sx} {sy} {self._maze_height / 2 * self._maze_unit}',
                    pos=f'{x} {y} {self._maze_height / 2 * self._maze_unit}',
                    material=f"door_{item['requires_key_color']}_material",
                    contype='1',
                    conaffinity='1',
                )

            for item in self._switch_items:
                x, y = item['xy']
                ET.SubElement(
                    worldbody,
                    'geom',
                    name=item['geom_name'],
                    type='cylinder',
                    # Keep switch as a thin floor region so the agent can pass over it.
                    size=f'{0.22 * self._maze_unit} {0.01 * self._maze_unit}',
                    pos=f'{x} {y} {0.01 * self._maze_unit}',
                    material=f"switch_{item['color']}_material",
                    contype='0',
                    conaffinity='0',
                )

            for item in self._gate_items:
                x, y = item['xy']
                sx, sy = item['door_size_xy']
                ET.SubElement(
                    worldbody,
                    'geom',
                    name=item['geom_name'],
                    type='box',
                    size=f'{sx} {sy} {self._maze_height / 2 * self._maze_unit}',
                    pos=f'{x} {y} {self._maze_height / 2 * self._maze_unit}',
                    material='gate_black_material',
                    contype='1',
                    conaffinity='1',
                )

            # Adjust floor size.
            maze_h, maze_w = self.maze_map.shape
            center_x = 0.5 * self._maze_unit * (maze_w - 1) - self._offset_x
            center_y = 0.5 * self._maze_unit * (maze_h - 1) - self._offset_y
            size_x = 0.5 * self._maze_unit * maze_w
            size_y = 0.5 * self._maze_unit * maze_h
            floor = tree.find('.//geom[@name="floor"]')
            floor.set('pos', f'{center_x} {center_y} 0')
            floor.set('size', f'{size_x} {size_y} 0.2')

            if self._teleport_info is not None:
                # Add teleports.
                for i, (x, y) in enumerate(self._teleport_info['teleport_in_xys']):
                    ET.SubElement(
                        worldbody,
                        'geom',
                        name=f'teleport_in_{i}',
                        type='cylinder',
                        size=f'{self._teleport_info["teleport_radius"]} .05',
                        pos=f'{x} {y} .05',
                        material='teleport_in',
                        contype='0',
                        conaffinity='0',
                    )
                for i, (x, y) in enumerate(self._teleport_info['teleport_out_xys']):
                    ET.SubElement(
                        worldbody,
                        'geom',
                        name=f'teleport_out_{i}',
                        type='cylinder',
                        size=f'{self._teleport_info["teleport_radius"]} .05',
                        pos=f'{x} {y} .05',
                        material='teleport_out',
                        contype='0',
                        conaffinity='0',
                    )

            if self._ob_type == 'pixels':
                # Color wall.
                wall = tree.find('.//material[@name="wall"]')
                wall.set('rgba', '.6 .6 .6 1')
                # Remove ambient light.
                light = tree.find('.//light[@name="global"]')
                light.attrib.pop('ambient')
                # Remove torso light.
                torso_light = tree.find('.//light[@name="torso_light"]')
                torso_light_parent = tree.find('.//light[@name="torso_light"]/..')
                torso_light_parent.remove(torso_light)
                # Remove texture repeat.
                grid = tree.find('.//material[@name="grid"]')
                grid.set('texuniform', 'false')
                if loco_env_type == 'ant':
                    # Color one leg white to break symmetry.
                    tree.find('.//geom[@name="aux_1_geom"]').set('material', 'self_white')
                    tree.find('.//geom[@name="left_leg_geom"]').set('material', 'self_white')
                    tree.find('.//geom[@name="left_ankle_geom"]').set('material', 'self_white')
            else:
                # Only show the target for states-based observation.
                ET.SubElement(
                    worldbody,
                    'geom',
                    name='target',
                    type='box',
                    size='.45 .45 .05',
                    pos='0 0 .05',
                    material='target',
                    rgba='.20 .82 .28 1',
                    contype='0',
                    conaffinity='0',
                )

        def set_tasks(self):
            # `tasks` is a list of tasks, where each task is a list of two tuples: (init_ij, goal_ij).
            if self._maze_type == 'arena':
                tasks = [
                    [(1, 1), (6, 6)],
                ]
            elif self._maze_type == 'medium':
                tasks = [
                    [(1, 1), (6, 6)],
                    [(6, 1), (1, 6)],
                    [(5, 3), (4, 2)],
                    [(6, 5), (6, 1)],
                    [(2, 6), (1, 1)],
                ]
            elif self._maze_type == 'large':
                tasks = [
                    [(1, 1), (7, 10)],
                    [(5, 4), (7, 1)],
                    [(7, 4), (1, 10)],
                    [(3, 8), (5, 4)],
                    [(1, 1), (5, 4)],
                ]
            elif self._maze_type == 'giant':
                tasks = [
                    [(1, 1), (10, 14)],
                    [(1, 14), (10, 1)],
                    [(8, 14), (1, 1)],
                    [(8, 3), (5, 12)],
                    [(5, 9), (3, 8)],
                ]
            elif self._maze_type == 'teleport':
                tasks = [
                    [(1, 10), (7, 1)],
                    [(1, 1), (7, 10)],
                    [(5, 6), (7, 10)],
                    [(7, 1), (7, 10)],
                    [(5, 6), (7, 1)],
                ]
            elif self._maze_type == 'multipath':
                tasks = [
                    [(5, 1), (5, 13)],
                    [(1, 1), (5, 13)],
                    [(9, 1), (5, 13)],
                    [(5, 1), (1, 12)],
                    [(5, 1), (9, 12)],
                ]
            elif self._maze_type in ('windingcorridor', 'longpath'):
                tasks = [
                    [(1, 1), (7, 1)],
                    [(7, 1), (1, 1)],
                    [(1, 4), (7, 12)],
                    [(3, 10), (7, 3)],
                    [(5, 3), (1, 15)],
                ]
            elif self._maze_type == 'custom':
                if self._custom_tasks is not None:
                    tasks = self._custom_tasks
                else:
                    free_ijs = np.argwhere(self.maze_map == 0)
                    if len(free_ijs) < 2:
                        raise ValueError('Custom maze must have at least two free cells to define a task.')
                    init_ij = tuple(int(v) for v in free_ijs[0])
                    goal_ij = tuple(int(v) for v in free_ijs[-1])
                    tasks = [[init_ij, goal_ij]]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self.task_infos = []
            for i, task in enumerate(tasks):
                self.task_infos.append(
                    dict(
                        task_name=f'task{i + 1}',
                        init_ij=task[0],
                        init_xy=self.ij_to_xy(task[0]),
                        goal_ij=task[1],
                        goal_xy=self.ij_to_xy(task[1]),
                    )
                )

            if self._reward_task_id == 0:
                self._reward_task_id = 1  # Default task.

        def initialize_renderer(self):
            # Make custom renderer.
            self.custom_renderer = mujoco.Renderer(
                self.model,
                width=self.width,
                height=self.height,
            )
            self.render()

        def reset(self, options=None, *args, **kwargs):
            if options is None:
                options = {}
            # Set the task goal.
            if self._reward_task_id is not None:
                # Use the pre-defined task.
                assert 1 <= self._reward_task_id <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = self._reward_task_id
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_id' in options:
                # Use the pre-defined task.
                assert 1 <= options['task_id'] <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = options['task_id']
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_info' in options:
                # Use the provided task information.
                self.cur_task_id = None
                self.cur_task_info = options['task_info']
            else:
                # Randomly sample a task.
                self.cur_task_id = np.random.randint(1, self.num_tasks + 1)
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]

            # Whether to provide a rendering of the goal.
            render_goal = False
            if 'render_goal' in options:
                render_goal = options['render_goal']

            # Get initial and goal positions with noise.
            init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['init_ij']))
            goal_xy = self.ij_to_xy(self.cur_task_info['goal_ij'])
            if self._add_noise_to_goal:
                goal_xy = self.add_noise(goal_xy)

            # Ensure positions are valid free-space cells for point-based agents.
            if loco_env_type in ('point', 'pointy'):
                if not self._is_xy_in_free_cell(init_xy):
                    init_xy = self.ij_to_xy(self.cur_task_info['init_ij'])
                if not self._is_xy_in_free_cell(goal_xy):
                    goal_xy = self.ij_to_xy(self.cur_task_info['goal_ij'])

            # First, force set the position to the goal position to obtain the goal observation.
            super().reset(*args, **kwargs)

            # Do a few random steps to stabilize the environment.
            num_random_actions = 40 if loco_env_type == 'humanoid' else 5
            for _ in range(num_random_actions):
                super().step(self._move_action_space.sample())

            # Save the goal observation.
            self.set_goal(goal_xy=goal_xy)
            self.set_xy(goal_xy)
            goal_ob = self.get_oracle_rep() if self._use_oracle_rep else self.get_ob()
            if render_goal:
                goal_rendered = self.render()

            # Now, do the actual reset.
            ob, info = super().reset(*args, **kwargs)
            self.set_goal(goal_xy=goal_xy)
            self.set_xy(init_xy)
            self._reset_key_door_mechanisms()
            ob = self.get_ob()
            info['goal'] = goal_ob
            if render_goal:
                info['goal_rendered'] = goal_rendered

            return ob, info

        def step(self, action):
            if self._success_timing == 'pre':
                success = self.compute_success()

            move_action, interaction_cmd = self._split_action(action)

            prev_xy = self.get_xy().copy() if loco_env_type in ('point', 'pointy') else None

            ob, reward, terminated, truncated, info = super().step(move_action)

            # Keep point-based agents from crossing walls or leaving the maze bounds.
            movement_blocked = False
            if loco_env_type in ('point', 'pointy'):
                cur_xy = self.get_xy().copy()

                # Strict collision rule: if the swept agent disk would overlap any wall,
                # closed door, or closed gate, reject the whole translation.
                if prev_xy is not None:
                    if not self._is_motion_collision_free(prev_xy, cur_xy):
                        self.set_xy(prev_xy)
                        ob = self.get_ob()
                        movement_blocked = True
                elif not self._is_xy_in_free_cell(cur_xy):
                    self.set_xy(prev_xy)
                    ob = self.get_ob()
                    movement_blocked = True

            if self._success_timing == 'post':
                success = self.compute_success()

            if self._teleport_info is not None:
                # Check if the agent is close to a inbound teleport.
                for x, y in self._teleport_info['teleport_in_xys']:
                    if np.linalg.norm(self.get_xy() - np.array([x, y])) <= self._teleport_info['teleport_radius'] * 1.5:
                        # Teleport the agent to a random outbound teleport.
                        teleport_out_xy = self._teleport_info['teleport_out_xys'][
                            np.random.randint(len(self._teleport_info['teleport_out_xys']))
                        ]
                        self.set_xy(np.array(teleport_out_xy))
                        break

            key_events = self._update_key_door_mechanisms(interaction_cmd=interaction_cmd)
            if key_events['picked_key_ids']:
                info['picked_key_ids'] = key_events['picked_key_ids']
            if key_events['opened_door_ids']:
                info['opened_door_ids'] = key_events['opened_door_ids']
            if key_events['closed_door_ids']:
                info['closed_door_ids'] = key_events['closed_door_ids']
            if key_events['toggled_switch_ids']:
                info['toggled_switch_ids'] = key_events['toggled_switch_ids']
            if key_events['opened_gate_ids']:
                info['opened_gate_ids'] = key_events['opened_gate_ids']
            if key_events['closed_gate_ids']:
                info['closed_gate_ids'] = key_events['closed_gate_ids']
            if movement_blocked:
                info['movement_blocked'] = True

            if success:
                if self._terminate_at_goal:
                    terminated = True
                info['success'] = 1.0
                reward = 1.0
            else:
                info['success'] = 0.0
                reward = 0.0

            # If the environment is in the single-task mode, modify the reward.
            if self._reward_task_id is not None:
                reward = reward - 1.0  # -1 (failure) or 0 (success).

            return ob, reward, terminated, truncated, info

        def render(self):
            if self.render_mode == 'human':
                # Delegate to the parent MuJoCo env renderer so a window is shown.
                return super().render()

            if self.custom_renderer is None:
                self.initialize_renderer()
            self.custom_renderer.update_scene(self.data, camera=self.custom_camera)
            return self.custom_renderer.render()

        def get_ob(self, ob_type=None):
            ob_type = self._ob_type if ob_type is None else ob_type
            if ob_type == 'states':
                return super().get_ob()
            else:
                frame = self.render()
                return frame

        def get_oracle_rep(self):
            """Return the oracle goal representation (i.e., the goal position)."""
            return np.array(self.cur_goal_xy)

        def compute_success(self):
            if np.linalg.norm(self.get_xy() - self.cur_goal_xy) <= self._goal_tol:
                return True
            else:
                return False

        def set_goal(self, goal_ij=None, goal_xy=None):
            """Set the goal position and update the target object."""
            if goal_xy is None:
                self.cur_goal_xy = self.ij_to_xy(goal_ij)
                if self._add_noise_to_goal:
                    self.cur_goal_xy = self.add_noise(self.cur_goal_xy)
            else:
                self.cur_goal_xy = goal_xy
            if self._ob_type == 'states':
                self.model.geom('target').pos[:2] = goal_xy

        def get_oracle_subgoal(self, start_xy, goal_xy):
            """Get the oracle subgoal for the agent.

            If the goal is unreachable, it returns the current position as the subgoal.

            Args:
                start_xy: Starting position of the agent.
                goal_xy: Goal position of the agent.
            Returns:
                A tuple of the oracle subgoal and the BFS map.
            """
            start_ij = self.xy_to_ij(start_xy)
            goal_ij = self.xy_to_ij(goal_xy)

            # Run BFS to find the next subgoal.
            bfs_map = self.maze_map.copy()
            for i in range(self.maze_map.shape[0]):
                for j in range(self.maze_map.shape[1]):
                    bfs_map[i][j] = -1

            bfs_map[goal_ij[0], goal_ij[1]] = 0
            queue = [goal_ij]
            while len(queue) > 0:
                i, j = queue.pop(0)
                for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < self.maze_map.shape[0]
                        and 0 <= nj < self.maze_map.shape[1]
                        and self.maze_map[ni, nj] == 0
                        and bfs_map[ni, nj] == -1
                    ):
                        bfs_map[ni][nj] = bfs_map[i][j] + 1
                        queue.append((ni, nj))

            # Find the subgoal that attains the minimum BFS value.
            subgoal_ij = start_ij
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = start_ij[0] + di, start_ij[1] + dj
                if (
                    0 <= ni < self.maze_map.shape[0]
                    and 0 <= nj < self.maze_map.shape[1]
                    and self.maze_map[ni, nj] == 0
                    and bfs_map[ni, nj] < bfs_map[subgoal_ij[0], subgoal_ij[1]]
                ):
                    subgoal_ij = (ni, nj)
            subgoal_xy = self.ij_to_xy(subgoal_ij)
            return np.array(subgoal_xy), bfs_map

        def xy_to_ij(self, xy):
            maze_unit = self._maze_unit
            i = int((xy[1] + self._offset_y + 0.5 * maze_unit) / maze_unit)
            j = int((xy[0] + self._offset_x + 0.5 * maze_unit) / maze_unit)
            return i, j

        def ij_to_xy(self, ij):
            i, j = ij
            x = j * self._maze_unit - self._offset_x
            y = i * self._maze_unit - self._offset_y
            return x, y

        def add_noise(self, xy):
            random_x = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
            random_y = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
            return xy[0] + random_x, xy[1] + random_y

        def _is_xy_in_free_cell(self, xy):
            i, j = self.xy_to_ij(xy)
            if i < 0 or i >= self.maze_map.shape[0] or j < 0 or j >= self.maze_map.shape[1]:
                return False
            return self.maze_map[i, j] == 0

        def _is_motion_path_clear(self, start_xy, end_xy):
            """Return True if the swept center path stays in free cells.

            This prevents point agents from cutting corners through walls when
            a single action moves across a cell boundary.
            """
            start_xy = np.asarray(start_xy, dtype=np.float32)
            end_xy = np.asarray(end_xy, dtype=np.float32)
            delta = end_xy - start_xy
            dist = float(np.linalg.norm(delta))
            if dist <= 1e-8:
                return True

            # Sample along the segment at sub-cell resolution.
            step_len = max(0.05 * self._maze_unit, 1e-4)
            num_samples = max(2, int(np.ceil(dist / step_len)) + 1)
            for t in np.linspace(0.0, 1.0, num_samples)[1:]:
                probe_xy = start_xy + t * delta
                if not self._is_xy_in_free_cell(probe_xy):
                    return False
            return True

        def _get_point_agent_radius(self):
            """Get point/pointy collision radius in world units."""
            return float(self._point_collision_radius)

        def _iter_blocking_rects(self):
            """Yield axis-aligned blocking rectangles as (cx, cy, hx, hy)."""
            wall_half = 0.5 * self._maze_unit
            for i in range(self.maze_map.shape[0]):
                for j in range(self.maze_map.shape[1]):
                    if self.maze_map[i, j] == 1:
                        cx, cy = self.ij_to_xy((i, j))
                        yield float(cx), float(cy), float(wall_half), float(wall_half)

            for door in self._door_items:
                if bool(door.get('opened', False)):
                    continue
                cx, cy = door['closed_xy']
                hx, hy = door['door_size_xy']
                yield float(cx), float(cy), float(hx), float(hy)

            for gate in self._gate_items:
                if bool(gate.get('opened', False)):
                    continue
                cx, cy = gate['closed_xy']
                hx, hy = gate['door_size_xy']
                yield float(cx), float(cy), float(hx), float(hy)

        def _disk_overlaps_rect(self, xy, radius, rect):
            """Return True when a disk centered at xy overlaps an axis-aligned rect."""
            cx, cy, hx, hy = rect
            dx = abs(float(xy[0]) - cx) - (hx + radius)
            dy = abs(float(xy[1]) - cy) - (hy + radius)
            return dx <= 0.0 and dy <= 0.0

        def _disk_clearance_to_rect(self, xy, radius, rect):
            """Return non-negative clearance from disk boundary to rectangle boundary."""
            cx, cy, hx, hy = rect
            dx = abs(float(xy[0]) - cx) - (hx + radius)
            dy = abs(float(xy[1]) - cy) - (hy + radius)
            ox = max(dx, 0.0)
            oy = max(dy, 0.0)
            return float(np.hypot(ox, oy))

        def _is_motion_collision_free(self, start_xy, end_xy):
            """Return True if swept disk from start_xy to end_xy avoids all blockers."""
            start_xy = np.asarray(start_xy, dtype=np.float32)
            end_xy = np.asarray(end_xy, dtype=np.float32)
            delta = end_xy - start_xy
            dist = float(np.linalg.norm(delta))

            radius = self._get_point_agent_radius()
            blocking_rects = list(self._iter_blocking_rects())

            # If already near contact, disallow moves that press further into blockers.
            contact_margin = 0.03 * self._maze_unit
            clearance_eps = 1e-5
            for rect in blocking_rects:
                start_clearance = self._disk_clearance_to_rect(start_xy, radius, rect)
                if start_clearance > contact_margin:
                    continue
                end_clearance = self._disk_clearance_to_rect(end_xy, radius, rect)
                if end_clearance + clearance_eps < start_clearance:
                    return False

            if dist <= 1e-8:
                for rect in blocking_rects:
                    if self._disk_overlaps_rect(start_xy, radius, rect):
                        return False
                return True

            # Fine enough sampling to avoid tunneling through thin doors.
            step_len = max(0.02 * self._maze_unit, 0.05 * max(radius, 1e-4))
            num_samples = max(3, int(np.ceil(dist / step_len)) + 1)
            for t in np.linspace(0.0, 1.0, num_samples):
                probe_xy = start_xy + t * delta
                for rect in blocking_rects:
                    if self._disk_overlaps_rect(probe_xy, radius, rect):
                        return False
            return True

        def _split_action(self, action):
            if self._has_open_action:
                # Preferred format: (move_action, cmd), where cmd is discrete:
                # 0=no-op, 1=pickup, 2=open/use.
                if isinstance(action, (tuple, list)) and len(action) == 2:
                    move_action = np.asarray(action[0], dtype=np.float32).reshape(-1)
                    if move_action.size < 2:
                        move_action = np.zeros(2, dtype=np.float32)
                    else:
                        move_action = move_action[:2]

                    interact_raw = action[1]
                    if isinstance(interact_raw, (np.ndarray, list, tuple)):
                        interact_arr = np.asarray(interact_raw).reshape(-1)
                        interact_val = int(interact_arr[0]) if interact_arr.size > 0 else 0
                    else:
                        interact_val = int(interact_raw)

                    if interact_val < 0:
                        interact_val = 0
                    if interact_val > 2:
                        interact_val = 2
                    return move_action, interact_val

            # Backward-compatible fallback for old continuous 3-vector actions.
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if self._has_open_action and arr.size >= 3:
                # Map previous boolean interact to open/use.
                return arr[:2], 2 if bool(arr[2] >= 0.5) else 0
            if arr.size >= 2:
                return arr[:2], 0
            # Fallback for malformed scalar actions.
            return np.zeros(2, dtype=np.float32), 0

        def _initialize_key_door_mechanisms(self):
            self._key_items = []
            self._door_items = []
            self._switch_items = []
            self._gate_items = []
            self._inventory_key_colors = set()

            if not isinstance(self._raw_mechanisms, dict):
                return

            keys = self._raw_mechanisms.get('keys', [])
            doors = self._raw_mechanisms.get('doors', [])
            switches = self._raw_mechanisms.get('switches', [])
            gates = self._raw_mechanisms.get('gates', [])
            if keys is None:
                keys = []
            if doors is None:
                doors = []
            if switches is None:
                switches = []
            if gates is None:
                gates = []

            key_color_by_id = {}
            for idx, key in enumerate(keys):
                if not isinstance(key, dict) or 'position' not in key:
                    continue
                color = self._normalize_color_name(key.get('color', 'yellow'))
                key_id = str(key.get('id', f'k{idx}'))
                key_color_by_id[key_id] = color
                i, j = int(key['position'][1]), int(key['position'][0])
                self._key_items.append(
                    dict(
                        id=key_id,
                        color=color,
                        ij=(i, j),
                        xy=self.ij_to_xy((i, j)),
                        collected=False,
                        geom_names=[
                            f'json_key_ring_{idx}_{self._sanitize_name(key_id)}',
                            f'json_key_stem_{idx}_{self._sanitize_name(key_id)}',
                            f'json_key_tooth_{idx}_{self._sanitize_name(key_id)}',
                        ],
                        geom_ids=[],
                    )
                )

            for idx, door in enumerate(doors):
                if not isinstance(door, dict) or 'position' not in door:
                    continue
                door_id = str(door.get('id', f'd{idx}'))
                required = door.get('requires_key', 'yellow')
                required = key_color_by_id.get(str(required), required)
                required_color = self._normalize_color_name(required)
                initial_state = str(door.get('initial_state', 'locked')).lower()
                locked = initial_state != 'open'
                i, j = int(door['position'][1]), int(door['position'][0])
                closed_xy, door_size_xy, slide_dir_xy = self._compute_door_geometry(i, j)
                open_xy = (
                    closed_xy[0] + slide_dir_xy[0] * 1.0 * self._maze_unit,
                    closed_xy[1] + slide_dir_xy[1] * 1.0 * self._maze_unit,
                )
                self._door_items.append(
                    dict(
                        id=door_id,
                        requires_key_color=required_color,
                        ij=(i, j),
                        xy=closed_xy,
                        closed_xy=closed_xy,
                        open_xy=open_xy,
                        door_size_xy=door_size_xy,
                        locked=locked,
                        initial_locked=locked,
                        opened=False,
                        geom_name=f'json_door_{idx}_{self._sanitize_name(door_id)}',
                        geom_id=None,
                    )
                )

            gate_by_id = {}
            gate_controlled_by = {}
            for idx, gate in enumerate(gates):
                if not isinstance(gate, dict) or 'position' not in gate:
                    continue
                gate_id = str(gate.get('id', f'g{idx}'))
                controlled_by = gate.get('controlled_by', [])
                if isinstance(controlled_by, str):
                    controlled_by = [controlled_by]
                elif not isinstance(controlled_by, (list, tuple, set)):
                    controlled_by = []
                gate_controlled_by[gate_id] = [str(sid) for sid in controlled_by]
                initial_state = str(gate.get('initial_state', 'closed')).lower()
                closed = initial_state != 'open'
                i, j = int(gate['position'][1]), int(gate['position'][0])
                closed_xy, door_size_xy, slide_dir_xy = self._compute_door_geometry(i, j)
                open_xy = (
                    closed_xy[0] + slide_dir_xy[0] * 1.0 * self._maze_unit,
                    closed_xy[1] + slide_dir_xy[1] * 1.0 * self._maze_unit,
                )
                gate_item = dict(
                    id=gate_id,
                    ij=(i, j),
                    xy=closed_xy,
                    closed_xy=closed_xy,
                    open_xy=open_xy,
                    door_size_xy=door_size_xy,
                    closed=closed,
                    opened=not closed,
                    initial_opened=not closed,
                    geom_name=f'json_gate_{idx}_{self._sanitize_name(gate_id)}',
                    geom_id=None,
                )
                self._gate_items.append(gate_item)
                gate_by_id[gate_id] = gate_item

            default_switch_type = str(self._mechanism_rules.get('switch_type', 'toggle')).lower()
            for idx, switch in enumerate(switches):
                if not isinstance(switch, dict) or 'position' not in switch:
                    continue
                switch_id = str(switch.get('id', f's{idx}'))
                i, j = int(switch['position'][1]), int(switch['position'][0])
                controls = switch.get('controls', [])
                if not isinstance(controls, list):
                    controls = []
                controls = [str(cid) for cid in controls if str(cid) in gate_by_id]

                # Also support reverse JSON wiring where each gate declares `controlled_by`.
                for gate_id, controller_ids in gate_controlled_by.items():
                    if switch_id in controller_ids:
                        controls.append(gate_id)

                # Keep deterministic ordering and remove duplicates.
                controls = list(dict.fromkeys(controls))
                switch_type = str(switch.get('switch_type', default_switch_type)).lower()
                initial_state = str(switch.get('initial_state', 'off')).lower()
                color = self._normalize_color_name(switch.get('color', 'yellow'))
                is_on = initial_state in {'on', 'true', '1'}
                self._switch_items.append(
                    dict(
                        id=switch_id,
                        ij=(i, j),
                        xy=self.ij_to_xy((i, j)),
                        controls=controls,
                        switch_type=switch_type,
                        color=color,
                        is_on=is_on,
                        initial_is_on=is_on,
                        geom_name=f'json_switch_{idx}_{self._sanitize_name(switch_id)}',
                        geom_id=None,
                    )
                )

        def _bind_key_door_geoms(self):
            for item in self._key_items:
                item['geom_ids'] = []
                for geom_name in item['geom_names']:
                    try:
                        item['geom_ids'].append(self.model.geom(geom_name).id)
                    except Exception:
                        continue

            for item in self._door_items:
                try:
                    item['geom_id'] = self.model.geom(item['geom_name']).id
                except Exception:
                    item['geom_id'] = None

            for item in self._switch_items:
                try:
                    item['geom_id'] = self.model.geom(item['geom_name']).id
                except Exception:
                    item['geom_id'] = None

            for item in self._gate_items:
                try:
                    item['geom_id'] = self.model.geom(item['geom_name']).id
                except Exception:
                    item['geom_id'] = None

        def _reset_key_door_mechanisms(self):
            self._inventory_key_colors = set()

            for item in self._key_items:
                item['collected'] = False
                for geom_id in item['geom_ids']:
                    self.model.geom_rgba[geom_id, 3] = 1.0

            for item in self._door_items:
                locked = bool(item.get('initial_locked', item.get('locked', True)))
                item['locked'] = locked
                item['opened'] = False
                geom_id = item['geom_id']
                if geom_id is None:
                    continue
                if locked:
                    self._set_door_closed_geom(geom_id, item['closed_xy'])
                else:
                    self._set_door_open_geom(geom_id, item['open_xy'])
                    item['opened'] = True

            for item in self._switch_items:
                item['is_on'] = bool(item.get('initial_is_on', False))
                geom_id = item['geom_id']
                if geom_id is None:
                    continue
                self.model.geom_rgba[geom_id, :4] = self._switch_rgba(item)

            for gate in self._gate_items:
                gate['opened'] = bool(gate.get('initial_opened', False))
                gate['closed'] = not gate['opened']

            self._refresh_gates_from_switches()

            mujoco.mj_forward(self.model, self.data)

        def _set_gate_closed_geom(self, geom_id, closed_xy):
            self.model.geom_pos[geom_id, :3] = np.array(
                [closed_xy[0], closed_xy[1], self._maze_height / 2 * self._maze_unit]
            )
            self.model.geom_contype[geom_id] = 1
            self.model.geom_conaffinity[geom_id] = 1
            self.model.geom_rgba[geom_id, 3] = 1.0

        def _set_gate_open_geom(self, geom_id, open_xy):
            # Hide open gates to avoid partial-panel artifacts in narrow corridors.
            self.model.geom_pos[geom_id, :3] = np.array([open_xy[0], open_xy[1], -10.0 * self._maze_unit])
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
            self.model.geom_rgba[geom_id, 3] = 0.0

        def _refresh_gates_from_switches(self):
            for gate in self._gate_items:
                controllers = [s for s in self._switch_items if gate['id'] in s['controls']]
                if len(controllers) > 0:
                    should_open = any(s['is_on'] for s in controllers)
                else:
                    should_open = gate['opened']

                geom_id = gate['geom_id']
                if should_open:
                    gate['opened'] = True
                    gate['closed'] = False
                    if geom_id is not None:
                        self._set_gate_open_geom(geom_id, gate['open_xy'])
                else:
                    gate['opened'] = False
                    gate['closed'] = True
                    if geom_id is not None:
                        self._set_gate_closed_geom(geom_id, gate['closed_xy'])

        def _set_door_closed_geom(self, geom_id, closed_xy):
            self.model.geom_pos[geom_id, :3] = np.array(
                [closed_xy[0], closed_xy[1], self._maze_height / 2 * self._maze_unit]
            )
            self.model.geom_contype[geom_id] = 1
            self.model.geom_conaffinity[geom_id] = 1
            self.model.geom_rgba[geom_id, 3] = 1.0

        def _set_door_open_geom(self, geom_id, open_xy):
            # Hide open doors to avoid partial-panel artifacts in narrow corridors.
            self.model.geom_pos[geom_id, :3] = np.array([open_xy[0], open_xy[1], -10.0 * self._maze_unit])
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
            self.model.geom_rgba[geom_id, 3] = 0.0

        def _update_key_door_mechanisms(self, interaction_cmd=0):
            events = {
                'picked_key_ids': [],
                'opened_door_ids': [],
                'closed_door_ids': [],
                'toggled_switch_ids': [],
                'opened_gate_ids': [],
                'closed_gate_ids': [],
            }
            if (
                len(self._key_items) == 0
                and len(self._door_items) == 0
                and len(self._switch_items) == 0
                and len(self._gate_items) == 0
            ):
                return events

            cur_xy = self.get_xy()
            wants_pickup = interaction_cmd == 1
            wants_use = interaction_cmd == 2

            # Pick up keys.
            if wants_pickup:
                for item in self._key_items:
                    if item['collected']:
                        continue
                    if np.linalg.norm(cur_xy - np.array(item['xy'])) <= self._key_pickup_radius:
                        item['collected'] = True
                        self._inventory_key_colors.add(item['color'])
                        events['picked_key_ids'].append(item['id'])
                        for geom_id in item['geom_ids']:
                            self.model.geom_rgba[geom_id, 3] = 0.0

            # Toggle doors when the agent explicitly chooses to use and is close enough.
            for item in self._door_items:
                if not wants_use:
                    continue
                if not self._is_near_door_for_interaction(cur_xy, item):
                    continue

                geom_id = item['geom_id']

                # If door is currently open, interact closes it.
                if item['opened']:
                    if geom_id is not None:
                        self._set_door_closed_geom(geom_id, item['closed_xy'])
                    item['opened'] = False
                    events['closed_door_ids'].append(item['id'])
                    continue

                # Closed door can be opened if already unlocked or matching key is in inventory.
                can_open = (not bool(item.get('locked', True))) or (
                    item['requires_key_color'] in self._inventory_key_colors
                )
                if can_open:
                    if geom_id is not None:
                        self._set_door_open_geom(geom_id, item['open_xy'])
                    item['opened'] = True
                    item['locked'] = False
                    if bool(self._mechanism_rules.get('key_consumption', True)) and (
                        item['requires_key_color'] in self._inventory_key_colors
                    ):
                        self._inventory_key_colors.discard(item['requires_key_color'])
                    events['opened_door_ids'].append(item['id'])

            # Toggle switches with the interaction action when close.
            if wants_use:
                for item in self._switch_items:
                    if np.linalg.norm(cur_xy - np.array(item['xy'])) > self._switch_toggle_radius:
                        continue
                    if item['switch_type'] == 'toggle':
                        item['is_on'] = not item['is_on']
                    else:
                        item['is_on'] = True
                    events['toggled_switch_ids'].append(item['id'])
                    geom_id = item['geom_id']
                    if geom_id is not None:
                        self.model.geom_rgba[geom_id, :4] = self._switch_rgba(item)

            # Gate state follows switch states.
            prev_gate_open = {g['id']: bool(g['opened']) for g in self._gate_items}
            self._refresh_gates_from_switches()
            for gate in self._gate_items:
                was_open = prev_gate_open.get(gate['id'], False)
                is_open = bool(gate['opened'])
                if (not was_open) and is_open:
                    events['opened_gate_ids'].append(gate['id'])
                elif was_open and (not is_open):
                    events['closed_gate_ids'].append(gate['id'])

            if (
                events['picked_key_ids']
                or events['opened_door_ids']
                or events['closed_door_ids']
                or events['toggled_switch_ids']
                or events['opened_gate_ids']
                or events['closed_gate_ids']
            ):
                mujoco.mj_forward(self.model, self.data)
            return events

        def _is_near_door_for_interaction(self, cur_xy, door_item):
            """Return True if the agent is close enough to interact with a door panel.

            Distance is measured to the door rectangle boundary (not door center),
            which is robust for long doors that span a corridor.
            """
            cx, cy = door_item['closed_xy']
            sx, sy = door_item['door_size_xy']

            dx = abs(float(cur_xy[0]) - float(cx)) - float(sx)
            dy = abs(float(cur_xy[1]) - float(cy)) - float(sy)
            outside = np.maximum(np.array([dx, dy], dtype=np.float32), 0.0)
            dist_to_panel = float(np.linalg.norm(outside))
            return dist_to_panel <= float(self._door_open_radius)

        def _sanitize_name(self, value):
            return re.sub(r'[^a-zA-Z0-9_\-]', '_', str(value))

        def _normalize_color_name(self, color):
            if color is None:
                return 'yellow'
            color_name = str(color).strip().lower()
            if color_name == 'grey':
                return 'gray'
            if color_name == 'purple':
                return 'magenta'
            if color_name not in {
                'red',
                'green',
                'blue',
                'yellow',
                'orange',
                'cyan',
                'magenta',
                'white',
                'black',
                'gray',
            }:
                return 'yellow'
            return color_name

        def _rgba_for_color(self, color_name, alpha=1.0):
            palette = {
                'red': (0.90, 0.22, 0.22),
                'green': (0.20, 0.78, 0.30),
                'blue': (0.22, 0.45, 0.90),
                'yellow': (0.95, 0.85, 0.20),
                'orange': (0.95, 0.55, 0.20),
                'cyan': (0.20, 0.85, 0.85),
                'magenta': (0.78, 0.30, 0.90),
                'white': (0.95, 0.95, 0.95),
                'black': (0.15, 0.15, 0.15),
                'gray': (0.55, 0.55, 0.55),
            }
            color_name = self._normalize_color_name(color_name)
            r, g, b = palette[color_name]
            return f'{r:.3f} {g:.3f} {b:.3f} {float(alpha):.3f}'

        def _rgba_array_for_color(self, color_name, alpha=1.0, brightness=1.0):
            rgba = np.array([float(v) for v in self._rgba_for_color(color_name, alpha=alpha).split()], dtype=np.float32)
            rgba[:3] = np.clip(rgba[:3] * float(brightness), 0.0, 1.0)
            return rgba

        def _switch_rgba(self, item):
            brightness = 1.0 if item.get('is_on', False) else 0.72
            return self._rgba_array_for_color(item.get('color', 'yellow'), alpha=1.0, brightness=brightness)

        def _compute_door_geometry(self, i, j):
            """Infer a wall-attached closed door shape and its slide direction."""

            def is_wall(ii, jj):
                if ii < 0 or ii >= self.maze_map.shape[0] or jj < 0 or jj >= self.maze_map.shape[1]:
                    return True
                return self.maze_map[ii, jj] == 1

            def wall_distance_cells(di, dj):
                steps = 1
                while True:
                    ii = i + di * steps
                    jj = j + dj * steps
                    if is_wall(ii, jj):
                        return max(0.5, steps - 0.5)
                    steps += 1

            base_x, base_y = self.ij_to_xy((i, j))
            dist_up = wall_distance_cells(-1, 0) * self._maze_unit
            dist_down = wall_distance_cells(1, 0) * self._maze_unit
            dist_left = wall_distance_cells(0, -1) * self._maze_unit
            dist_right = wall_distance_cells(0, 1) * self._maze_unit

            # Door thickness across travel direction; span attaches to opposite walls.
            thickness = 0.09 * self._maze_unit
            min_half_span = 0.12 * self._maze_unit

            vertical_span = dist_up + dist_down
            horizontal_span = dist_left + dist_right

            # Horizontal corridor (agent mostly moves in x): door spans y (touches top/bottom walls).
            if vertical_span <= horizontal_span:
                half_span_y = max(min_half_span, 0.5 * vertical_span)
                center_shift_y = 0.5 * (dist_down - dist_up)
                closed_xy = (base_x, base_y + center_shift_y)
                door_size_xy = (thickness, half_span_y)
                # Slide along corridor travel direction (x), toward the closer side wall.
                slide_dir_xy = (-1.0 if dist_left <= dist_right else 1.0, 0.0)
                return closed_xy, door_size_xy, slide_dir_xy

            # Vertical corridor (agent mostly moves in y): door spans x (touches left/right walls).
            half_span_x = max(min_half_span, 0.5 * horizontal_span)
            center_shift_x = 0.5 * (dist_right - dist_left)
            closed_xy = (base_x + center_shift_x, base_y)
            door_size_xy = (half_span_x, thickness)
            # Slide along corridor travel direction (y), toward the closer side wall.
            slide_dir_xy = (0.0, -1.0 if dist_up <= dist_down else 1.0)
            return closed_xy, door_size_xy, slide_dir_xy

    class BallEnv(MazeEnv):
        def update_tree(self, tree):
            super().update_tree(tree)

            # Add ball.
            worldbody = tree.find('.//worldbody')
            ball = ET.SubElement(worldbody, 'body', name='ball', pos='0 0 0.5')
            ET.SubElement(ball, 'freejoint', name='ball_root')
            ET.SubElement(
                ball,
                'geom',
                name='ball',
                size='.25',
                material='ball',
                priority='1',
                conaffinity='1',
                condim='6',
            )
            ET.SubElement(ball, 'light', name='ball_light', pos='0 0 4', mode='trackcom')

        def set_tasks(self):
            # `tasks` is a list of tasks, where each task is a list of three tuples: (agent_init_ij, ball_init_ij,
            # goal_ij).
            if self._maze_type == 'arena':
                tasks = [
                    [(1, 6), (2, 3), (5, 2)],
                    [(2, 2), (5, 5), (2, 2)],
                    [(6, 1), (2, 3), (6, 6)],
                    [(6, 6), (1, 1), (6, 1)],
                    [(4, 6), (6, 2), (1, 6)],
                ]
            elif self._maze_type == 'medium':
                tasks = [
                    [(1, 1), (3, 4), (6, 6)],
                    [(6, 1), (6, 5), (1, 1)],
                    [(5, 3), (4, 2), (6, 5)],
                    [(6, 5), (1, 1), (5, 3)],
                    [(1, 6), (6, 1), (1, 6)],
                ]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self.task_infos = []
            for i, task in enumerate(tasks):
                self.task_infos.append(
                    dict(
                        task_name=f'task{i + 1}',
                        agent_init_ij=task[0],
                        agent_init_xy=self.ij_to_xy(task[0]),
                        ball_init_ij=task[1],
                        ball_init_xy=self.ij_to_xy(task[1]),
                        goal_ij=task[2],
                        goal_xy=self.ij_to_xy(task[2]),
                    )
                )

            if self._reward_task_id == 0:
                self._reward_task_id = 4  # Default task.

        def reset(self, options=None, *args, **kwargs):
            if options is None:
                options = {}
            # Set the task goal.
            if self._reward_task_id is not None:
                # Use the pre-defined task.
                assert 1 <= self._reward_task_id <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = self._reward_task_id
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_id' in options:
                # Use the pre-defined task.
                assert 1 <= options['task_id'] <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = options['task_id']
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_info' in options:
                # Use the provided task information.
                self.cur_task_id = None
                self.cur_task_info = options['task_info']
            else:
                # Randomly sample a task.
                self.cur_task_id = np.random.randint(1, self.num_tasks + 1)
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]

            # Whether to provide a rendering of the goal.
            render_goal = False
            if 'render_goal' in options:
                render_goal = options['render_goal']

            # Get initial and goal positions with noise.
            agent_init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['agent_init_ij']))
            ball_init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['ball_init_ij']))
            goal_xy = self.ij_to_xy(self.cur_task_info['goal_ij'])
            if self._add_noise_to_goal:
                goal_xy = self.add_noise(goal_xy)

            # First, force set the position to the goal position to obtain the goal observation.
            super(MazeEnv, self).reset(*args, **kwargs)

            # Do a few random steps to stabilize the environment.
            for _ in range(10):
                super(MazeEnv, self).step(self.action_space.sample())

            # Save the goal observation.
            self.set_goal(goal_xy=goal_xy)
            self.set_agent_ball_xy(goal_xy, goal_xy)
            goal_ob = self.get_oracle_rep() if self._use_oracle_rep else self.get_ob()
            if render_goal:
                goal_rendered = self.render()

            # Now, do the actual reset.
            ob, info = super(MazeEnv, self).reset(*args, **kwargs)
            self.set_goal(goal_xy=goal_xy)
            self.set_agent_ball_xy(agent_init_xy, ball_init_xy)
            ob = self.get_ob()
            info['goal'] = goal_ob
            if render_goal:
                info['goal_rendered'] = goal_rendered

            return ob, info

        def step(self, action):
            if self._success_timing == 'pre':
                success = self.compute_success()

            ob, reward, terminated, truncated, info = super(MazeEnv, self).step(action)

            if self._success_timing == 'post':
                success = self.compute_success()

            if success:
                if self._terminate_at_goal:
                    terminated = True
                info['success'] = 1.0
                reward = 1.0
            else:
                info['success'] = 0.0
                reward = 0.0

            # If the environment is in the single-task mode, modify the reward.
            if self._reward_task_id is not None:
                reward = reward - 1.0  # -1 (failure) or 0 (success).

            return ob, reward, terminated, truncated, info

        def compute_success(self):
            if np.linalg.norm(self.get_agent_ball_xy()[1] - self.cur_goal_xy) <= self._goal_tol:
                return True
            else:
                return False

        def get_agent_ball_xy(self):
            agent_xy = self.data.qpos[:2].copy()
            ball_xy = self.data.qpos[-7:-5].copy()

            return agent_xy, ball_xy

        def set_agent_ball_xy(self, agent_xy, ball_xy):
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
            qpos[:2] = agent_xy
            qpos[-7:-5] = ball_xy
            self.set_state(qpos, qvel)

    if maze_env_type == 'maze':
        return MazeEnv(*args, **kwargs)
    elif maze_env_type == 'ball':
        return BallEnv(*args, **kwargs)
    else:
        raise ValueError(f'Unknown maze environment type: {maze_env_type}')
