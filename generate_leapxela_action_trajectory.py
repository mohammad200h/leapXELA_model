"""Generate LeapXELA grasp action trajectories for sim and hardware.

This is an offline utility: it writes arrays to disk and never commands motors.
Simulation actions use the MuJoCo actuator order from leapXela_base_model.xml.
Hardware actions use motor-ID order from joint_config.json.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
DEFAULT_XML_PATH = _SCRIPT_DIR / "leapXela_base_model.xml"
DEFAULT_JOINT_CONFIG = _SCRIPT_DIR / "joint_config.json"
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "leapxela_action_trajectory.npz"

CLOSE_START = 0.0
CLOSE_END = 1.5
PREGRIP_FRACTION = 0.40
THUMB_GRIP_FRACTION = 0.35
THUMB_DELAY = 0.5
GRASP_PATTERNS = ("hold", "pulse", "squeeze", "regrasp", "tap", "shear")
SQUEEZE_STEP_SECONDS = 0.9
REGRASP_HZ = 0.35
TAP_HZ = 1.5
TAP_DEPTH = 0.5
SHEAR_HZ = 0.7
EXPECTED_JOINT_COUNT = 16
LIMIT_TOLERANCE = 1e-8
UNIFORM_SCALE_RTOL = 1e-3


@dataclass(frozen=True)
class GraspProfile:
    pattern: str
    grip_fraction: float
    thumb_grip_fraction: float
    pregrip_fraction: float
    thumb_delay: float
    pulse_hz: float
    pulse_amplitude: float
    shear_amplitude: float
    squeeze_steps: int


def parse_float_pair(text: str) -> tuple[float, float]:
    values = [float(item) for item in text.split()]
    if len(values) != 2:
        raise ValueError(f"Expected two floats, got '{text}'")
    return values[0], values[1]


def load_sim_actuators(xml_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    root = ET.parse(xml_path).getroot()

    ctrlrange_by_class = {}
    for default in root.findall(".//default"):
        if "class" not in default.attrib:
            continue
        position = default.find("position")
        if position is None or "ctrlrange" not in position.attrib:
            continue
        ctrlrange_by_class[default.attrib["class"]] = parse_float_pair(
            position.attrib["ctrlrange"]
        )

    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError(f"No <actuator> block found in {xml_path}")

    actuator_names = []
    joint_names = []
    ctrlranges = []
    for position in actuator.findall("position"):
        for field in ("name", "joint", "class"):
            if field not in position.attrib:
                raise ValueError(f"Actuator position is missing '{field}'")
        actuator_class = position.attrib["class"]
        if actuator_class not in ctrlrange_by_class:
            raise ValueError(f"No ctrlrange default for class '{actuator_class}'")
        actuator_names.append(position.attrib["name"])
        joint_names.append(position.attrib["joint"])
        ctrlranges.append(ctrlrange_by_class[actuator_class])

    validate_joint_names(joint_names, "simulation")
    return actuator_names, joint_names, np.asarray(ctrlranges, dtype=np.float64)


def validate_joint_names(joint_names: list[str], label: str) -> None:
    if len(joint_names) != EXPECTED_JOINT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_JOINT_COUNT} {label} joints, got {len(joint_names)}"
        )
    if len(set(joint_names)) != len(joint_names):
        raise ValueError(f"{label} joint names are not unique: {joint_names}")


def load_joint_config(joint_config_path: Path) -> dict:
    return json.loads(joint_config_path.read_text())


def hardware_joint_names(joint_config: dict) -> list[str]:
    hardware_map = joint_config["leapXela"]["hardware"]["map"]
    indexed_names = []
    for finger in sorted(hardware_map):
        for index_text, joint_name in hardware_map[finger].items():
            indexed_names.append((int(index_text), f"{finger}_{joint_name}"))

    indexed_names.sort(key=lambda item: item[0])
    indices = [item[0] for item in indexed_names]
    expected_indices = list(range(EXPECTED_JOINT_COUNT))
    if indices != expected_indices:
        raise ValueError(f"Hardware map indices are {indices}, expected {expected_indices}")

    names = [item[1] for item in indexed_names]
    validate_joint_names(names, "hardware")
    return names


def ranges_for_joint_names(joint_names: list[str], joint_config: dict) -> dict[str, np.ndarray]:
    ranges = {"ll": [], "ul": [], "zero": []}
    for full_name in joint_names:
        prefix, joint_name = full_name.split("_", 1)
        section_name = "thumb" if prefix == "th" else "fingers"
        section = joint_config[section_name]
        ranges["ll"].append(float(section[joint_name]["lower"]))
        ranges["ul"].append(float(section[joint_name]["upper"]))
        if "zero" in section[joint_name]:
            ranges["zero"].append(float(section[joint_name]["zero"]))
        else:
            ranges["zero"].append(0.0)
    return {key: np.asarray(value, dtype=np.float64) for key, value in ranges.items()}


def pregrip_targets(ctrlrange: np.ndarray, pregrip_fraction: float) -> np.ndarray:
    low = ctrlrange[:, 0]
    high = ctrlrange[:, 1]
    return low + pregrip_fraction * (high - low)


def control_targets(
    ctrlrange: np.ndarray, joint_names: list[str], t: float, profile: GraspProfile
) -> np.ndarray:
    low = ctrlrange[:, 0]
    high = ctrlrange[:, 1]
    span = high - low
    is_thumb = np.asarray([name.startswith("th_") for name in joint_names], dtype=bool)

    grip = low + np.where(
        is_thumb, profile.thumb_grip_fraction, profile.grip_fraction
    ) * span
    pregrip = pregrip_targets(ctrlrange, profile.pregrip_fraction)

    close_span = CLOSE_END - CLOSE_START
    finger_phase = np.clip((t - CLOSE_START) / close_span, 0.0, 1.0)
    thumb_phase = np.clip(
        (t - CLOSE_START - profile.thumb_delay) / close_span, 0.0, 1.0
    )
    phase = np.where(is_thumb, thumb_phase, finger_phase)
    smooth = phase * phase * (3.0 - 2.0 * phase)
    targets = pregrip + smooth * (grip - pregrip)
    if t < CLOSE_END + profile.thumb_delay:
        return np.clip(targets, low, high)

    elapsed = t - CLOSE_END - profile.thumb_delay
    if profile.pattern == "hold":
        modulation = 0.0
    elif profile.pattern == "pulse":
        modulation = 0.5 * profile.pulse_amplitude * (
            1.0 - np.cos(2.0 * np.pi * profile.pulse_hz * elapsed)
        )
    elif profile.pattern == "squeeze":
        cycle = profile.squeeze_steps * 2
        index = int(elapsed / SQUEEZE_STEP_SECONDS) % cycle
        level = index if index < profile.squeeze_steps else cycle - index - 1
        modulation = profile.pulse_amplitude * level / max(
            profile.squeeze_steps - 1, 1
        )
    elif profile.pattern == "regrasp":
        release = 0.5 * (1.0 - np.cos(2.0 * np.pi * REGRASP_HZ * elapsed))
        return np.clip(targets + release * (pregrip - targets), low, high)
    elif profile.pattern == "tap":
        release = 0.5 * (1.0 - np.cos(2.0 * np.pi * TAP_HZ * elapsed))
        return np.clip(targets + TAP_DEPTH * release * (pregrip - targets), low, high)
    elif profile.pattern == "shear":
        lateral = profile.shear_amplitude * np.sin(2.0 * np.pi * SHEAR_HZ * elapsed)
        is_lateral = np.asarray(
            [name.endswith("_rot") or name.endswith("_axl") for name in joint_names],
            dtype=bool,
        )
        return np.clip(targets + np.where(is_lateral, lateral * span, 0.0), low, high)
    else:
        raise ValueError(f"Unknown grasp pattern '{profile.pattern}'")

    return np.clip(targets + np.where(is_thumb, 0.0, modulation * span), low, high)


def generate_times(duration: float, frequency: float) -> np.ndarray:
    sample_count = int(np.floor(duration * frequency)) + 1
    return np.arange(sample_count, dtype=np.float64) / frequency


def generate_sim_actions(
    times: np.ndarray, ctrlrange: np.ndarray, joint_names: list[str], profile: GraspProfile
) -> np.ndarray:
    actions = [
        control_targets(ctrlrange, joint_names, float(t), profile)
        for t in times
    ]
    return np.asarray(actions, dtype=np.float64)


def reorder_actions(
    actions: np.ndarray, source_names: list[str], target_names: list[str]
) -> np.ndarray:
    source_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_index]
    if missing:
        raise ValueError(f"Missing source joints for hardware order: {missing}")
    order = [source_index[name] for name in target_names]
    return actions[:, order]


def piecewise_map(
    value: float,
    sim_ll: float,
    sim_zero: float,
    sim_ul: float,
    hw_ll: float,
    hw_zero: float,
    hw_ul: float,
    index: int,
) -> float:
    if value <= sim_zero:
        denom = sim_zero - sim_ll
        if denom == 0.0:
            if abs(value - sim_zero) < 1e-12:
                return hw_zero
            raise ValueError(
                f"Invalid sim range at index {index}: lower == zero == {sim_ll}"
            )
        alpha = (value - sim_ll) / denom
        return alpha * (hw_zero - hw_ll) + hw_ll

    denom = sim_ul - sim_zero
    if denom == 0.0:
        if abs(value - sim_zero) < 1e-12:
            return hw_zero
        raise ValueError(
            f"Invalid sim range at index {index}: zero == upper == {sim_ul}"
        )
    alpha = (value - sim_zero) / denom
    return alpha * (hw_ul - hw_zero) + hw_zero


def has_uniform_scale(
    sim_ll: float,
    sim_zero: float,
    sim_ul: float,
    hw_ll: float,
    hw_zero: float,
    hw_ul: float,
    rtol: float,
) -> bool:
    low_sim = sim_zero - sim_ll
    low_hw = abs(hw_zero - hw_ll)
    if low_sim == 0.0:
        if low_hw != 0.0:
            return False
    elif abs(low_hw / low_sim - 1.0) > rtol:
        return False

    high_sim = sim_ul - sim_zero
    high_hw = abs(hw_ul - hw_zero)
    if high_sim == 0.0:
        if high_hw != 0.0:
            return False
    elif abs(high_hw / high_sim - 1.0) > rtol:
        return False

    return True


def uniform_offset_map(value: float, hw_zero: float, inverted: bool) -> float:
    if inverted:
        return hw_zero - value
    return hw_zero + value


def map_sim_to_hardware(
    sim_actions: np.ndarray, sim_ranges: dict[str, np.ndarray],
    hardware_ranges: dict[str, np.ndarray],
) -> np.ndarray:
    hardware_actions = np.zeros_like(sim_actions)
    for row_index in range(sim_actions.shape[0]):
        for joint_index in range(sim_actions.shape[1]):
            value = float(sim_actions[row_index, joint_index])
            sim_ll = float(sim_ranges["ll"][joint_index])
            sim_ul = float(sim_ranges["ul"][joint_index])
            sim_zero = float(sim_ranges["zero"][joint_index])
            hw_ll = float(hardware_ranges["ll"][joint_index])
            hw_ul = float(hardware_ranges["ul"][joint_index])
            hw_zero = float(hardware_ranges["zero"][joint_index])

            if abs(sim_zero) < 1e-9 and has_uniform_scale(
                sim_ll, sim_zero, sim_ul, hw_ll, hw_zero, hw_ul, UNIFORM_SCALE_RTOL
            ):
                hardware_actions[row_index, joint_index] = uniform_offset_map(
                    value, hw_zero, hw_ll > hw_ul
                )
            else:
                hardware_actions[row_index, joint_index] = piecewise_map(
                    value, sim_ll, sim_zero, sim_ul, hw_ll, hw_zero, hw_ul,
                    joint_index,
                )
    return hardware_actions


def validate_values_finite(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains NaN or infinite values")


def validate_values_in_ranges(
    values: np.ndarray, ranges: np.ndarray, joint_names: list[str], label: str
) -> None:
    lower = np.minimum(ranges[:, 0], ranges[:, 1]) - LIMIT_TOLERANCE
    upper = np.maximum(ranges[:, 0], ranges[:, 1]) + LIMIT_TOLERANCE
    below = values < lower
    above = values > upper
    if not np.any(below | above):
        return

    rows, cols = np.where(below | above)
    first_row = int(rows[0])
    first_col = int(cols[0])
    name = joint_names[first_col]
    value = float(values[first_row, first_col])
    raise ValueError(
        f"{label} value {value:.6f} for {name} at row {first_row} is outside "
        f"[{lower[first_col]:.6f}, {upper[first_col]:.6f}]"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.frequency <= 0.0:
        raise ValueError("--frequency must be positive")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if args.squeeze_steps <= 0:
        raise ValueError("--squeeze-steps must be positive")


def profile_from_args(args: argparse.Namespace) -> GraspProfile:
    return GraspProfile(
        pattern=args.grasp_pattern,
        grip_fraction=args.grip_fraction,
        thumb_grip_fraction=args.thumb_grip_fraction,
        pregrip_fraction=args.pregrip_fraction,
        thumb_delay=args.thumb_delay,
        pulse_hz=args.pulse_hz,
        pulse_amplitude=args.pulse_amplitude,
        shear_amplitude=args.shear_amplitude,
        squeeze_steps=args.squeeze_steps,
    )


def save_trajectory(
    output_path: Path,
    times: np.ndarray,
    sim_actions: np.ndarray,
    hardware_actions: np.ndarray,
    sim_joint_names: list[str],
    hardware_names: list[str],
    actuator_names: list[str],
    profile: GraspProfile,
    frequency: float,
    duration: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        times=times,
        sim_actions=sim_actions,
        hardware_actions=hardware_actions,
        sim_joint_names=np.asarray(sim_joint_names),
        hardware_joint_names=np.asarray(hardware_names),
        actuator_names=np.asarray(actuator_names),
        frequency_hz=np.asarray(frequency, dtype=np.float64),
        duration=np.asarray(duration, dtype=np.float64),
        grasp_pattern=np.asarray(profile.pattern),
        profile_json=np.asarray(json.dumps(asdict(profile), sort_keys=True)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate offline LeapXELA sim and hardware action arrays."
    )
    parser.add_argument("-f", "--frequency", type=float, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--grasp-pattern", choices=GRASP_PATTERNS, default="pulse",
    )
    parser.add_argument("--grip-fraction", type=float, default=0.55)
    parser.add_argument("--thumb-grip-fraction", type=float, default=THUMB_GRIP_FRACTION)
    parser.add_argument("--pregrip-fraction", type=float, default=PREGRIP_FRACTION)
    parser.add_argument("--thumb-delay", type=float, default=THUMB_DELAY)
    parser.add_argument("--pulse-hz", type=float, default=1.0)
    parser.add_argument("--pulse-amplitude", type=float, default=0.15)
    parser.add_argument("--shear-amplitude", type=float, default=0.15)
    parser.add_argument("--squeeze-steps", type=int, default=4)
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--joint-config", type=Path, default=DEFAULT_JOINT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    actuator_names, sim_joint_names, ctrlrange = load_sim_actuators(args.xml_path)
    joint_config = load_joint_config(args.joint_config)
    hardware_names = hardware_joint_names(joint_config)

    profile = profile_from_args(args)
    times = generate_times(args.duration, args.frequency)
    sim_actions = generate_sim_actions(times, ctrlrange, sim_joint_names, profile)
    validate_values_finite(sim_actions, "sim_actions")
    validate_values_in_ranges(sim_actions, ctrlrange, sim_joint_names, "sim_actions")

    sim_actions_hw_order = reorder_actions(sim_actions, sim_joint_names, hardware_names)
    leapxela_config = joint_config["leapXela"]
    sim_ranges_hw_order = ranges_for_joint_names(hardware_names, leapxela_config["sim"])
    hardware_ranges = ranges_for_joint_names(hardware_names, leapxela_config["hardware"])
    validate_values_in_ranges(
        sim_actions_hw_order,
        np.column_stack((sim_ranges_hw_order["ll"], sim_ranges_hw_order["ul"])),
        hardware_names,
        "sim_actions in conversion order",
    )

    hardware_actions = map_sim_to_hardware(
        sim_actions_hw_order, sim_ranges_hw_order, hardware_ranges
    )
    validate_values_finite(hardware_actions, "hardware_actions")
    validate_values_in_ranges(
        hardware_actions,
        np.column_stack((hardware_ranges["ll"], hardware_ranges["ul"])),
        hardware_names,
        "hardware_actions",
    )

    save_trajectory(
        args.output,
        times,
        sim_actions,
        hardware_actions,
        sim_joint_names,
        hardware_names,
        actuator_names,
        profile,
        args.frequency,
        args.duration,
    )

    print(f"Wrote {args.output}")
    print(f"times: {times.shape}")
    print(f"sim_actions: {sim_actions.shape} order={sim_joint_names}")
    print(f"hardware_actions: {hardware_actions.shape} order={hardware_names}")
    print(
        "first sim action: "
        + np.array2string(sim_actions[0], precision=6, separator=", ")
    )
    print(
        "first hardware action: "
        + np.array2string(hardware_actions[0], precision=6, separator=", ")
    )


if __name__ == "__main__":
    main()
