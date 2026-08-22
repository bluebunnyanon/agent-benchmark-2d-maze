import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from pathlib import Path

from ogbench.procgen.maze_json_interface import make_pointymaze_env_from_json


# Specify the maze JSON file to visualize.
# Default: single_key example. You can override via first CLI arg.
default_json = Path(__file__).parent / 'ogbench' / 'procgen' / 'maze_jsons' / 'D2'/ '10x10_dense_wrong_ky_inactive_sb_sg_kr_0.json'
maze_json_path = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else default_json

# Build a pointymaze environment directly from the JSON maze.
env, parsed_maze = make_pointymaze_env_from_json(
	maze_json_path,
	json_origin='top_left',
	render_mode='rgb_array',
	width=512,
	height=512,
	ob_type='states',
	add_noise_to_goal=False,
)

print(f'Loaded maze: {maze_json_path}', flush=True)
print(f'Task ID: {parsed_maze.task_id}', flush=True)
print(f'Dimensions (W,H): ({parsed_maze.width}, {parsed_maze.height})', flush=True)
print(f'Mechanisms keys: {len(parsed_maze.mechanisms.get("keys", []))}', flush=True)
print(f'Mechanisms doors: {len(parsed_maze.mechanisms.get("doors", []))}', flush=True)
for key_obj in parsed_maze.mechanisms.get('keys', []):
	print(f'Key {key_obj.get("id", "?")} at {key_obj.get("position", "?")}', flush=True)
for door_obj in parsed_maze.mechanisms.get('doors', []):
	print(f'Door {door_obj.get("id", "?")} at {door_obj.get("position", "?")}', flush=True)
print(parsed_maze.to_ascii(add_boundary_walls=True, include_mechanisms=True), flush=True)
print(f'Action space: {env.action_space}', flush=True)
print('Legend: S=start, G=goal, K=key, D=door', flush=True)

# Render the pointymaze env.
# Disable Matplotlib default keymaps so game controls are not intercepted
# (e.g., 'p' pan, 's' save, 'f' fullscreen, arrow navigation).
for key in list(mpl.rcParams.keys()):
	if key.startswith('keymap.'):
		mpl.rcParams[key] = []

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
img_artist = None
status_artist = None
cursor_pos_artist = None

print('Keyboard controls:', flush=True)
print('  No key pressed: pause (do not step)', flush=True)
print('  Drive: Left/Right=delta heading, Up/Down=delta forward', flush=True)
print('  Interact: P=pickup, I=open/use', flush=True)
print('  Reset: R', flush=True)
print('  Quit: Q or Esc', flush=True)

episode_max_steps = 1000

pressed_keys = set()
quit_requested = False
reset_requested = False
move_scale = 1.0
pending_interact_cmd = 0


def on_key_press(event):
	global quit_requested, reset_requested, pending_interact_cmd
	if event.key is None:
		return
	key = event.key.lower()
	if key in {'q', 'escape'}:
		quit_requested = True
	elif key == 'r':
		reset_requested = True
	elif key == 'p':
		pending_interact_cmd = 1
	elif key == 'i':
		pending_interact_cmd = 2
	elif key in {'up', 'down', 'left', 'right'}:
		pressed_keys.add(key)


def on_key_release(event):
	if event.key is None:
		return
	key = event.key.lower()
	pressed_keys.discard(key)


def on_focus_lost(_event):
	# Key release events may be missed when focus changes; clear latched movement keys.
	pressed_keys.clear()


def on_mouse_move(event):
	if cursor_pos_artist is None:
		return
	if event.inaxes != ax or event.xdata is None or event.ydata is None or img_artist is None:
		cursor_pos_artist.set_text('')
		return

	frame_h, frame_w = img_artist.get_array().shape[:2]
	x_px = float(np.clip(event.xdata, 0, max(frame_w - 1, 0)))
	y_px = float(np.clip(event.ydata, 0, max(frame_h - 1, 0)))

	# Map pixel-space cursor location to discrete maze cell coordinates.
	maze_x = int(np.floor((x_px / max(frame_w, 1)) * parsed_maze.width))
	maze_y = int(np.floor((y_px / max(frame_h, 1)) * parsed_maze.height))
	maze_x = int(np.clip(maze_x, 0, parsed_maze.width - 1))
	maze_y = int(np.clip(maze_y, 0, parsed_maze.height - 1))

	cursor_pos_artist.set_text(f'cursor: [{maze_x}, {maze_y}]')


def get_keyboard_action(action_space):
	global pending_interact_cmd
	has_motion_input = bool(pressed_keys)
	has_interact_input = pending_interact_cmd != 0
	if not has_motion_input and not has_interact_input:
		return None

	# Base move action is (delta_forward, delta_heading).
	move_action = np.zeros(2, dtype=np.float32)

	throttle = 0.0
	if 'up' in pressed_keys:
		throttle += 1.0
	if 'down' in pressed_keys:
		throttle -= 1.0

	delta_heading = 0.0
	if 'left' in pressed_keys:
		delta_heading += 1.0
	if 'right' in pressed_keys:
		delta_heading -= 1.0

	move_action[0] = move_scale * throttle
	move_action[1] = move_scale * delta_heading

	if hasattr(action_space, 'spaces') and len(action_space.spaces) >= 2:
		move_space = action_space.spaces[0]
		interact_space = action_space.spaces[1]
		if hasattr(move_space, 'shape') and move_space.shape is not None:
			target_move = np.zeros(move_space.shape, dtype=np.float32)
			if target_move.ndim > 0 and target_move.shape[0] >= 2:
				target_move[0] = move_action[0]
				target_move[1] = move_action[1]
			else:
				target_move = move_action.copy()
		else:
			target_move = move_action.copy()

		interact_cmd = int(pending_interact_cmd)
		pending_interact_cmd = 0
		if hasattr(interact_space, 'n'):
			interact_cmd = max(0, min(interact_cmd, int(interact_space.n) - 1))
		return (target_move, interact_cmd)

	if hasattr(action_space, 'shape') and action_space.shape is not None:
		action = np.zeros(action_space.shape, dtype=np.float32)
		if action.ndim > 0 and action.shape[0] >= 2:
			action[0] = move_action[0]
			action[1] = move_action[1]
		return action

	return move_action


press_cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
release_cid = fig.canvas.mpl_connect('key_release_event', on_key_release)
leave_cid = fig.canvas.mpl_connect('figure_leave_event', on_focus_lost)
close_cid = fig.canvas.mpl_connect('close_event', on_focus_lost)
motion_cid = fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

ob, info = env.reset(options=dict(task_id=1, render_goal=True))
steps = 0
episode_idx = 1
blocked_flash_frames = 0

try:
	while not quit_requested:
		if reset_requested:
			ob, info = env.reset(options=dict(task_id=1, render_goal=True))
			steps = 0
			episode_idx += 1
			reset_requested = False

		frame = env.render()
		if img_artist is None:
			img_artist = ax.imshow(frame)
			ax.set_title(f'Custom PointyMaze: {parsed_maze.task_id} (episode {episode_idx})')
			ax.axis('off')
			status_artist = ax.text(
				0.02,
				0.98,
				'',
				transform=ax.transAxes,
				va='top',
				ha='left',
				fontsize=12,
				color='white',
				bbox=dict(facecolor='crimson', alpha=0.85, edgecolor='none', pad=4),
			)
			cursor_pos_artist = ax.text(
				0.98,
				0.02,
				'',
				transform=ax.transAxes,
				va='bottom',
				ha='right',
				fontsize=11,
				color='white',
				bbox=dict(facecolor='black', alpha=0.65, edgecolor='none', pad=3),
			)
		else:
			img_artist.set_data(frame)

		if status_artist is not None:
			if blocked_flash_frames > 0:
				status_artist.set_text('BLOCKED by collision')
				blocked_flash_frames -= 1
			else:
				status_artist.set_text('')

		fig.canvas.draw_idle()
		plt.pause(0.001)

		action = get_keyboard_action(env.action_space)
		if action is None:
			time.sleep(0.03)
			continue

		ob, reward, terminated, truncated, info = env.step(action)
		steps += 1
		if info.get('movement_blocked'):
			blocked_flash_frames = 10

		if info.get('picked_key_ids'):
			print(f'Picked up keys: {info["picked_key_ids"]}')
		if info.get('opened_door_ids'):
			print(f'Opened doors: {info["opened_door_ids"]}')
		if info.get('closed_door_ids'):
			print(f'Closed doors: {info["closed_door_ids"]}')
		if info.get('toggled_switch_ids'):
			print(f'Toggled switches: {info["toggled_switch_ids"]}')
		if info.get('opened_gate_ids'):
			print(f'Opened gates: {info["opened_gate_ids"]}')
		if info.get('closed_gate_ids'):
			print(f'Closed gates: {info["closed_gate_ids"]}')

		# Explain failed door interactions to make debugging easier.
		interact_cmd = action[1] if isinstance(action, tuple) and len(action) > 1 else 0
		if interact_cmd == 2 and not info.get('opened_door_ids') and not info.get('closed_door_ids'):
			u = env.unwrapped
			inv_keys = sorted(getattr(u, '_inventory_key_colors', set()))
			nearby_msg = 'no nearby locked door'
			for door in getattr(u, '_door_items', []):
				if bool(door.get('opened', False)):
					continue
				cur_xy = np.array(u.get_xy(), dtype=np.float32)
				cx, cy = door['closed_xy']
				sx, sy = door['door_size_xy']
				dx = abs(float(cur_xy[0]) - float(cx)) - float(sx)
				dy = abs(float(cur_xy[1]) - float(cy)) - float(sy)
				outside = np.maximum(np.array([dx, dy], dtype=np.float32), 0.0)
				dist = float(np.linalg.norm(outside))
				required = str(door.get('requires_key_color', 'unknown'))
				radius = float(getattr(u, '_door_open_radius', 0.0))
				if dist <= radius:
					if required in inv_keys:
						nearby_msg = f'near {door["id"]} (dist={dist:.2f}) but it still did not open'
					else:
						nearby_msg = (
							f'near {door["id"]} (dist={dist:.2f}) but missing required key "{required}"'
						)
					break
			print(f'Interact used but no door opened: inventory={inv_keys}, reason={nearby_msg}')

		if terminated or truncated or steps >= episode_max_steps:
			print(f'Episode done: success={info.get("success", 0.0)} steps={steps}')
			ob, info = env.reset(options=dict(task_id=1, render_goal=True))
			steps = 0
			episode_idx += 1

		time.sleep(0.03)
finally:
	fig.canvas.mpl_disconnect(press_cid)
	fig.canvas.mpl_disconnect(release_cid)
	fig.canvas.mpl_disconnect(leave_cid)
	fig.canvas.mpl_disconnect(close_cid)
	fig.canvas.mpl_disconnect(motion_cid)
	env.close()
	plt.tight_layout()
	plt.show()