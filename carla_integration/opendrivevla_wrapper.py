"""
OpenDriveVLA Wrapper for CARLA Integration

This module provides a simplified interface to load and run OpenDriveVLA model
for autonomous driving in CARLA simulator.

Key Concepts:
- OpenDriveVLA outputs WAYPOINTS (trajectory points), NOT direct vehicle controls
- Waypoints are in ego-vehicle coordinate frame: +X = forward, +Y = left
- PID controller converts waypoints to throttle/steer/brake commands

Input Requirements:
- 6 cameras in nuScenes format (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
                                CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT)
- Ego vehicle states (velocity, acceleration, steering)
- Mission command (forward, left, right)

Output:
- 6 waypoints representing planned trajectory for next 3 seconds (0.5s interval)
- Format: [(x1,y1), (x2,y2), ..., (x6,y6)] in meters

Author: Claude Code
Date: 2025-11-25
"""

import os
import sys
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image

# Add OpenDriveVLA to path
OPENDRIVEVLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, OPENDRIVEVLA_ROOT)


@dataclass
class EgoState:
    """Ego vehicle state information"""
    velocity_x: float  # m/s, forward velocity
    velocity_y: float  # m/s, lateral velocity
    yaw_rate: float    # rad/s, angular velocity
    acceleration_x: float  # m/s^2
    acceleration_y: float  # m/s^2
    steering: float    # steering angle (normalized or radians)
    speed: float       # m/s, heading speed


@dataclass
class HistoricalTrajectory:
    """Historical trajectory points (last 2 seconds)"""
    # 4 points at 0.5s intervals: [-2.0s, -1.5s, -1.0s, -0.5s]
    points: List[Tuple[float, float]]  # [(x, y), ...]


class MissionCommand:
    """High-level mission command"""
    FORWARD = "keep forward"
    LEFT = "turn left"
    RIGHT = "turn right"


class OpenDriveVLAWrapper:
    """
    Wrapper for OpenDriveVLA model inference

    Usage:
        wrapper = OpenDriveVLAWrapper(model_path="checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct")
        wrapper.load_model()

        # Prepare inputs
        ego_state = EgoState(...)
        history = HistoricalTrajectory(...)
        command = MissionCommand.FORWARD

        # Run inference (simplified mode without full nuScenes data)
        waypoints = wrapper.predict_trajectory_simple(ego_state, history, command)
    """

    def __init__(
        self,
        model_path: str = "checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        use_bf16: bool = True,
        use_fp16: bool = False,
    ):
        self.model_path = os.path.join(OPENDRIVEVLA_ROOT, model_path)
        self.device = device
        self.use_bf16 = use_bf16
        self.use_fp16 = use_fp16

        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.is_loaded = False

        # nuScenes camera order
        self.camera_names = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]

        # Image normalization (nuScenes format)
        self.img_norm_cfg = {
            'mean': [103.530, 116.280, 123.675],
            'std': [1.0, 1.0, 1.0],
        }

    def load_model(self):
        """Load the OpenDriveVLA model"""
        if self.is_loaded:
            print("Model already loaded")
            return

        print(f"Loading OpenDriveVLA model from: {self.model_path}")

        try:
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init

            disable_torch_init()

            # Model configuration
            llava_model_args = {
                "multimodal": True,
                "attn_implementation": "sdpa"  # Use SDPA for PyTorch 2.x
            }

            overwrite_config = {
                "image_aspect_ratio": "pad",
                "vision_tower_test_mode": True
            }
            llava_model_args["overwrite_config"] = overwrite_config

            # Load model
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                self.model_path,
                model_base=None,
                model_name="llava_qwen",
                device_map=self.device,
                **llava_model_args
            )

            # Set to eval mode
            self.model.eval()

            self.is_loaded = True
            print(f"Model loaded successfully on {self.device}")
            print(f"  Model type: {type(self.model).__name__}")
            print(f"  Context length: {self.context_len}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _format_ego_message(self, ego_state: EgoState, history: HistoricalTrajectory) -> str:
        """Format ego state and history into model input message"""
        # Ego states message
        ego_msg = (
            f"- Velocity (vx,vy): ({ego_state.velocity_x:.2f},{ego_state.velocity_y:.2f})"
            f" - Heading Angular Velocity (v_yaw): ({ego_state.yaw_rate:.2f})"
            f" - Acceleration (ax,ay): ({ego_state.acceleration_x:.2f},{ego_state.acceleration_y:.2f})"
            f" - Can Bus: (0.00,0.00)"  # Placeholder
            f" - Heading Speed: ({ego_state.speed:.2f})"
            f" - Steering: ({ego_state.steering:.2f})"
        )

        # Historical trajectory message
        pts = history.points
        if len(pts) >= 4:
            his_msg = f"[({pts[0][0]:.2f},{pts[0][1]:.2f}),({pts[1][0]:.2f},{pts[1][1]:.2f}),({pts[2][0]:.2f},{pts[2][1]:.2f}),({pts[3][0]:.2f},{pts[3][1]:.2f})]"
        else:
            # Pad with zeros if not enough history
            his_msg = "[(0.00,0.00),(0.00,0.00),(0.00,0.00),(0.00,0.00)]"

        return ego_msg, his_msg

    def _build_prompt(
        self,
        ego_state: EgoState,
        history: HistoricalTrajectory,
        command: str
    ) -> str:
        """Build the full prompt for the model"""
        from llava.constants import (
            DEFAULT_SCENE_START_TOKEN, DEFAULT_SCENE_TOKEN, DEFAULT_SCENE_END_TOKEN,
            DEFAULT_TRACK_START_TOKEN, DEFAULT_TRACK_TOKEN, DEFAULT_TRACK_END_TOKEN,
            DEFAULT_MAP_START_TOKEN, DEFAULT_MAP_TOKEN, DEFAULT_MAP_END_TOKEN,
            DEFAULT_TRAJ_TOKEN
        )

        ego_msg, his_msg = self._format_ego_message(ego_state, history)

        prompt = (
            f"Scene information: {DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}\n"
            f"Object-wise tracking information: {DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}\n"
            f"Map information: {DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}\n"
            f"Ego states: {ego_msg}\n"
            f"Historical trajectory (last 2 seconds): {his_msg}\n"
            f"Mission goal: {command}\n"
            f"Planning trajectory: {DEFAULT_TRAJ_TOKEN}"
        )

        return prompt

    def _parse_trajectory_output(self, output_text: str) -> List[Tuple[float, float]]:
        """Parse trajectory waypoints from model output

        Expected format: <traj_start>[(x1,y1),(x2,y2),...,(x6,y6)]<traj_end>
        """
        # Extract content between trajectory tokens
        pattern = r'\[\([-\d.]+,[-\d.]+\)(?:,\([-\d.]+,[-\d.]+\))*\]'
        match = re.search(pattern, output_text)

        if match:
            traj_str = match.group()
            # Parse individual waypoints
            point_pattern = r'\(([-\d.]+),([-\d.]+)\)'
            points = re.findall(point_pattern, traj_str)
            waypoints = [(float(x), float(y)) for x, y in points]
            return waypoints
        else:
            print(f"Warning: Could not parse trajectory from output: {output_text}")
            return []

    def predict_trajectory_simple(
        self,
        ego_state: EgoState,
        history: HistoricalTrajectory,
        command: str = MissionCommand.FORWARD
    ) -> List[Tuple[float, float]]:
        """
        Simplified trajectory prediction without full UniAD perception data.

        Note: This is a simplified interface that generates prompts without
        the full nuScenes/UniAD perception data. For full accuracy, use
        the complete inference pipeline with 6-camera images and UniAD features.

        Returns:
            List of 6 waypoints [(x1,y1), ..., (x6,y6)] in ego coordinates (meters)
            Each waypoint is 0.5s apart, covering 3 seconds of future trajectory
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build prompt
        prompt = self._build_prompt(ego_state, history, command)

        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # Generate
        with torch.inference_mode():
            dtype = torch.bfloat16 if self.use_bf16 else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=256,
                    num_beams=1,
                )

        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Parse waypoints
        waypoints = self._parse_trajectory_output(output_text)

        return waypoints

    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_path": self.model_path,
            "model_type": type(self.model).__name__,
            "device": str(self.device),
            "context_length": self.context_len,
            "precision": "bf16" if self.use_bf16 else "fp16" if self.use_fp16 else "fp32",
        }


class WaypointToPIDController:
    """
    Convert waypoints to vehicle control commands using PID control

    This controller takes waypoint predictions from OpenDriveVLA and converts
    them to CARLA vehicle controls (throttle, steer, brake).
    """

    def __init__(
        self,
        target_speed: float = 5.0,  # m/s
        kp_lateral: float = 1.0,
        kd_lateral: float = 0.1,
        kp_longitudinal: float = 1.0,
        ki_longitudinal: float = 0.1,
        max_throttle: float = 0.75,
        max_brake: float = 1.0,
        max_steer: float = 0.8,
    ):
        self.target_speed = target_speed
        self.kp_lateral = kp_lateral
        self.kd_lateral = kd_lateral
        self.kp_longitudinal = kp_longitudinal
        self.ki_longitudinal = ki_longitudinal
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.max_steer = max_steer

        # State for PID
        self._prev_lateral_error = 0.0
        self._integral_error = 0.0

    def compute_control(
        self,
        waypoints: List[Tuple[float, float]],
        current_speed: float,
    ) -> Tuple[float, float, float]:
        """
        Compute vehicle control from waypoints

        Args:
            waypoints: List of (x, y) in ego coordinates. +X = forward, +Y = left
            current_speed: Current vehicle speed in m/s

        Returns:
            (throttle, steer, brake) - all in range [0, 1] or [-1, 1] for steer
        """
        if not waypoints or len(waypoints) < 2:
            # No valid waypoints, stop
            return 0.0, 0.0, 1.0

        # Use first waypoint for steering
        target_x, target_y = waypoints[0]

        # Lateral control (steering)
        # Pure pursuit style: steer towards target point
        # Negative Y = turn right, Positive Y = turn left
        lateral_error = target_y  # Lateral offset to target

        # PD control for steering
        lateral_d = lateral_error - self._prev_lateral_error
        self._prev_lateral_error = lateral_error

        steer = self.kp_lateral * lateral_error + self.kd_lateral * lateral_d
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        # Longitudinal control (throttle/brake)
        # Estimate target speed from waypoint distance
        target_distance = np.sqrt(target_x**2 + target_y**2)
        target_speed = min(self.target_speed, target_distance * 2.0)  # Slow down near target

        speed_error = target_speed - current_speed
        self._integral_error += speed_error * 0.05  # dt ~ 0.05s at 20Hz
        self._integral_error = np.clip(self._integral_error, -10, 10)

        acceleration = (
            self.kp_longitudinal * speed_error +
            self.ki_longitudinal * self._integral_error
        )

        if acceleration > 0:
            throttle = np.clip(acceleration, 0, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-acceleration, 0, self.max_brake)

        return float(throttle), float(steer), float(brake)

    def reset(self):
        """Reset PID state"""
        self._prev_lateral_error = 0.0
        self._integral_error = 0.0


def main():
    """Test the wrapper"""
    print("=" * 60)
    print("OpenDriveVLA Wrapper Test")
    print("=" * 60)

    # Test model loading
    wrapper = OpenDriveVLAWrapper()

    try:
        wrapper.load_model()
        print("\nModel Info:")
        for k, v in wrapper.get_model_info().items():
            print(f"  {k}: {v}")

        # Test simple prediction
        ego = EgoState(
            velocity_x=5.0,
            velocity_y=0.0,
            yaw_rate=0.0,
            acceleration_x=0.0,
            acceleration_y=0.0,
            steering=0.0,
            speed=5.0
        )

        history = HistoricalTrajectory(points=[
            (-4.0, 0.0),
            (-3.0, 0.0),
            (-2.0, 0.0),
            (-1.0, 0.0)
        ])

        print("\nTesting simple trajectory prediction...")
        waypoints = wrapper.predict_trajectory_simple(ego, history, MissionCommand.FORWARD)
        print(f"Predicted waypoints: {waypoints}")

        # Test PID controller
        print("\nTesting PID controller...")
        controller = WaypointToPIDController()
        throttle, steer, brake = controller.compute_control(waypoints, 5.0)
        print(f"Control output: throttle={throttle:.3f}, steer={steer:.3f}, brake={brake:.3f}")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
