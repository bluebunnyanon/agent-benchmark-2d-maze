from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ogbench.procgen.maze_json_interface import make_pointymaze_env_from_json, parse_maze_json_file


def _compute_maze_unit(json_path: Path) -> float:
    parsed = parse_maze_json_file(json_path)
    max_dim = max(parsed.width, parsed.height)
    # For larger mazes, zoom out a bit so boundary/perimeter walls stay in frame.
    return 1.4 * min(1.0, 10.0 / float(max_dim))


def render_maze_json(
    json_path: Path,
    output_path: Path,
    width: int,
    height: int,
    warmup_frames: int,
) -> None:
    maze_unit = _compute_maze_unit(json_path)
    env, _ = make_pointymaze_env_from_json(
        json_path,
        json_origin='top_left',
        maze_unit=maze_unit,
        render_mode='rgb_array',
        width=width,
        height=height,
        ob_type='states',
        add_noise_to_goal=False,
    )

    try:
        env.reset(options=dict(task_id=1, render_goal=True))
        frame = env.render()
        # Warm up rendering like the interactive viewer loop before saving.
        for _ in range(max(0, warmup_frames - 1)):
            frame = env.render()
        plt.imsave(output_path, frame)
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Render all maze JSON files to PNG images.'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('ogbench/procgen/maze_jsons'),
        help='Directory containing maze JSON files.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('ogbench/procgen/maze_images'),
        help='Directory where PNG images will be saved.',
    )
    parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='Rendered image width in pixels.',
    )
    parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='Rendered image height in pixels.',
    )
    parser.add_argument(
        '--warmup-frames',
        type=int,
        default=4,
        help='Number of render() calls after reset before saving.',
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    maze_files = sorted(input_dir.rglob('*.json'))
    if not maze_files:
        print(f'No JSON files found in: {input_dir}')
        return 1

    failures: list[tuple[Path, str]] = []
    for json_path in maze_files:
        relative_path = json_path.relative_to(input_dir)
        output_path = output_dir / relative_path.with_suffix('.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            render_maze_json(
                json_path,
                output_path,
                args.width,
                args.height,
                args.warmup_frames,
            )
            print(f'OK: {relative_path} -> {output_path.relative_to(output_dir)}')
        except Exception as exc:  # noqa: BLE001
            failures.append((json_path, str(exc)))
            print(f'FAIL: {relative_path} ({exc})')

    print(f'Rendered {len(maze_files) - len(failures)}/{len(maze_files)} mazes to {output_dir}')
    if failures:
        print('Failed files:')
        for path, err in failures:
            print(f'  - {path.relative_to(input_dir)}: {err}')
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
