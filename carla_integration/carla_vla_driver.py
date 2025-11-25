#!/usr/bin/env python3
"""
CARLA VLA Driver - Autonomous Driving with OpenDriveVLA

This module integrates OpenDriveVLA with CARLA simulator for
end-to-end autonomous driving using vision-language-action model.

Architecture:
    CARLA (6 cameras) → OpenDriveVLA → Waypoints → PID Controller → Vehicle Control

Usage:
    python carla_vla_driver.py [--model-path PATH] [--target-speed SPEED]

Author: Claude Code
Date: 2025-11-25
"""

import os
import sys
import time
import math
import threading
import queue
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

# Add CARLA path
CARLA_EGG = os.path.expanduser("~/CARLA_UE5_0.10.0/Carla-0.10.0-Linux-Shipping/PythonAPI/carla/dist/carla-0.10.0-py3.10-linux-x86_64.egg")
if os.path.exists(CARLA_EGG):
    sys.path.insert(0, CARLA_EGG)

import carla

# OpenDriveVLA imports (from local package)
OPENDRIVEVLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, OPENDRIVEVLA_ROOT)

# Direct import to avoid torch dependency in mock mode
import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from waypoint_controller import (
    WaypointVehicleController,
    HistoryTracker,
    ControlCommand
)


# ===========================================
# nuScenes-style Camera Configuration
# ===========================================
NUSCENES_CAMERAS = {
    'CAM_FRONT': {
        'location': carla.Location(x=1.5, y=0.0, z=1.6),
        'rotation': carla.Rotation(pitch=0, yaw=0, roll=0),
        'fov': 70,
    },
    'CAM_FRONT_LEFT': {
        'location': carla.Location(x=1.0, y=-0.5, z=1.6),
        'rotation': carla.Rotation(pitch=0, yaw=-55, roll=0),
        'fov': 70,
    },
    'CAM_FRONT_RIGHT': {
        'location': carla.Location(x=1.0, y=0.5, z=1.6),
        'rotation': carla.Rotation(pitch=0, yaw=55, roll=0),
        'fov': 70,
    },
    'CAM_BACK': {
        'location': carla.Location(x=-2.0, y=0.0, z=1.6),
        'rotation': carla.Rotation(pitch=0, yaw=180, roll=0),
        'fov': 110,
    },
    'CAM_BACK_LEFT': {
        'location': carla.Location(x=-1.0, y=-0.5, z=1.6),
        'rotation': carla.Rotation(pitch=0, yaw=-110, roll=0),
        'fov': 70,
    },
    'CAM_BACK_RIGHT': {
        'location': carla.Location(x=-1.0, y=0.5, z=1.6),
        'rotation': carla.Rotation(pitch=0, yaw=110, roll=0),
        'fov': 70,
    },
}

# Image resolution (nuScenes uses 1600x900)
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900


@dataclass
class VLAConfig:
    """Configuration for VLA driver"""
    model_path: str = "checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct"
    target_speed: float = 5.0  # m/s (~18 km/h)
    inference_rate: float = 5.0  # Hz (inference every 0.2s)
    use_bf16: bool = True
    lateral_controller: str = "pure_pursuit"


class CameraManager:
    """
    Manages 6 cameras in nuScenes configuration

    Handles camera spawning, image capture, and synchronization.
    """

    def __init__(self, world: carla.World, vehicle: carla.Actor, image_width: int = 1600, image_height: int = 900):
        self.world = world
        self.vehicle = vehicle
        self.image_width = image_width
        self.image_height = image_height

        self.cameras: Dict[str, carla.Actor] = {}
        self.latest_images: Dict[str, np.ndarray] = {}
        self.image_timestamps: Dict[str, float] = {}
        self.locks: Dict[str, threading.Lock] = {}

        for name in NUSCENES_CAMERAS.keys():
            self.locks[name] = threading.Lock()
            self.latest_images[name] = None
            self.image_timestamps[name] = 0.0

    def spawn_cameras(self):
        """Spawn all 6 cameras attached to vehicle"""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_width))
        camera_bp.set_attribute('image_size_y', str(self.image_height))

        for name, config in NUSCENES_CAMERAS.items():
            # Set FOV
            camera_bp.set_attribute('fov', str(config['fov']))

            # Create transform
            transform = carla.Transform(
                config['location'],
                config['rotation']
            )

            # Spawn camera
            camera = self.world.spawn_actor(
                camera_bp,
                transform,
                attach_to=self.vehicle
            )

            # Set up callback
            camera.listen(lambda image, cam_name=name: self._camera_callback(image, cam_name))

            self.cameras[name] = camera
            print(f"  ✓ Spawned {name}")

    def _camera_callback(self, image: carla.Image, camera_name: str):
        """Handle incoming camera image"""
        # Convert to numpy array (BGRA format)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha, keep BGR

        with self.locks[camera_name]:
            self.latest_images[camera_name] = array.copy()
            self.image_timestamps[camera_name] = time.time()

    def get_all_images(self) -> Dict[str, np.ndarray]:
        """Get latest images from all cameras"""
        images = {}
        for name in NUSCENES_CAMERAS.keys():
            with self.locks[name]:
                if self.latest_images[name] is not None:
                    images[name] = self.latest_images[name].copy()
        return images

    def get_synchronized_images(self, max_time_diff: float = 0.1) -> Optional[Dict[str, np.ndarray]]:
        """
        Get synchronized images from all cameras

        Returns None if images are not synchronized within max_time_diff seconds.
        """
        timestamps = []
        images = {}

        for name in NUSCENES_CAMERAS.keys():
            with self.locks[name]:
                if self.latest_images[name] is None:
                    return None
                timestamps.append(self.image_timestamps[name])
                images[name] = self.latest_images[name].copy()

        # Check synchronization
        if max(timestamps) - min(timestamps) > max_time_diff:
            return None

        return images

    def create_composite_view(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create a composite view of all 6 cameras for visualization

        Layout:
          [FRONT_LEFT] [FRONT] [FRONT_RIGHT]
          [BACK_LEFT]  [BACK]  [BACK_RIGHT]
        """
        # Resize images to smaller size for display
        scale = 0.4
        h = int(self.image_height * scale)
        w = int(self.image_width * scale)

        resized = {}
        for name, img in images.items():
            resized[name] = cv2.resize(img, (w, h))

        # Create composite
        top_row = np.hstack([
            resized.get('CAM_FRONT_LEFT', np.zeros((h, w, 3), dtype=np.uint8)),
            resized.get('CAM_FRONT', np.zeros((h, w, 3), dtype=np.uint8)),
            resized.get('CAM_FRONT_RIGHT', np.zeros((h, w, 3), dtype=np.uint8)),
        ])

        bottom_row = np.hstack([
            resized.get('CAM_BACK_LEFT', np.zeros((h, w, 3), dtype=np.uint8)),
            resized.get('CAM_BACK', np.zeros((h, w, 3), dtype=np.uint8)),
            resized.get('CAM_BACK_RIGHT', np.zeros((h, w, 3), dtype=np.uint8)),
        ])

        composite = np.vstack([top_row, bottom_row])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = [
            ('FRONT_LEFT', (10, 30)),
            ('FRONT', (w + 10, 30)),
            ('FRONT_RIGHT', (2*w + 10, 30)),
            ('BACK_LEFT', (10, h + 30)),
            ('BACK', (w + 10, h + 30)),
            ('BACK_RIGHT', (2*w + 10, h + 30)),
        ]

        for label, pos in labels:
            cv2.putText(composite, label, pos, font, 0.6, (0, 255, 0), 2)

        return composite

    def destroy(self):
        """Destroy all cameras"""
        for name, camera in self.cameras.items():
            try:
                camera.stop()
                camera.destroy()
            except:
                pass
        self.cameras.clear()


class VehicleStateTracker:
    """
    Tracks vehicle state for VLA input

    Collects ego vehicle information needed for OpenDriveVLA inference.
    """

    def __init__(self, vehicle: carla.Actor):
        self.vehicle = vehicle
        self.history = HistoryTracker(history_seconds=2.0, sample_rate=2.0)

        # State buffers
        self._prev_velocity = None
        self._prev_time = time.time()
        self._accel_x = 0.0
        self._accel_y = 0.0

    def update(self):
        """Update vehicle state tracking"""
        try:
            transform = self.vehicle.get_transform()
            location = transform.location

            # Update history
            self.history.update(location.x, location.y)

            # Store for acceleration calculation
            velocity = self.vehicle.get_velocity()
            current_time = time.time()

            if self._prev_velocity is not None:
                dt = current_time - self._prev_time
                if dt > 0:
                    self._accel_x = (velocity.x - self._prev_velocity.x) / dt
                    self._accel_y = (velocity.y - self._prev_velocity.y) / dt
                else:
                    self._accel_x = 0.0
                    self._accel_y = 0.0
            else:
                self._accel_x = 0.0
                self._accel_y = 0.0

            self._prev_velocity = velocity
            self._prev_time = current_time

        except Exception as e:
            print(f"State tracker error: {e}")

    def get_ego_state(self) -> Dict:
        """Get current ego state for VLA input"""
        try:
            velocity = self.vehicle.get_velocity()
            transform = self.vehicle.get_transform()
            control = self.vehicle.get_control()
            angular_velocity = self.vehicle.get_angular_velocity()

            # Speed in m/s
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            # Transform velocity to ego frame
            yaw_rad = math.radians(transform.rotation.yaw)
            cos_yaw = math.cos(-yaw_rad)
            sin_yaw = math.sin(-yaw_rad)

            vx_ego = velocity.x * cos_yaw - velocity.y * sin_yaw
            vy_ego = velocity.x * sin_yaw + velocity.y * cos_yaw

            return {
                'velocity_x': vx_ego,
                'velocity_y': vy_ego,
                'yaw_rate': math.radians(angular_velocity.z),
                'acceleration_x': self._accel_x,
                'acceleration_y': self._accel_y,
                'steering': control.steer,
                'speed': speed,
                'location': {
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'z': transform.location.z,
                },
                'rotation': {
                    'pitch': transform.rotation.pitch,
                    'yaw': transform.rotation.yaw,
                    'roll': transform.rotation.roll,
                }
            }
        except Exception as e:
            print(f"Get ego state error: {e}")
            return None

    def get_historical_trajectory(self) -> List[Tuple[float, float]]:
        """Get historical trajectory in ego frame"""
        try:
            transform = self.vehicle.get_transform()
            return self.history.get_history_in_ego_frame(
                transform.location.x,
                transform.location.y,
                math.radians(transform.rotation.yaw)
            )
        except:
            return [(0.0, 0.0)] * 4


class MockVLAModel:
    """
    Mock VLA model for testing without full model loaded

    Generates simple waypoints based on current heading.
    Useful for testing the control pipeline.
    """

    def __init__(self):
        self.is_loaded = True

    def predict_trajectory(self, ego_state: Dict, history: List, command: str) -> List[Tuple[float, float]]:
        """Generate mock waypoints"""
        # Simple straight line with slight curve based on command
        waypoints = []
        x = 0.0
        y = 0.0

        for i in range(6):
            x += 2.0  # 2m forward per waypoint (0.5s at ~4m/s)

            if command == "turn left":
                y += 0.3 * (i + 1)  # Gradual left turn
            elif command == "turn right":
                y -= 0.3 * (i + 1)  # Gradual right turn
            # else keep forward

            waypoints.append((x, y))

        return waypoints


class VLADriver:
    """
    Main VLA Driver class

    Coordinates all components for autonomous driving:
    - CARLA connection and vehicle setup
    - Camera management
    - VLA model inference
    - Vehicle control
    """

    def __init__(self, config: VLAConfig):
        self.config = config

        # CARLA
        self.client = None
        self.world = None
        self.vehicle = None

        # Components
        self.camera_manager = None
        self.state_tracker = None
        self.controller = None
        self.vla_model = None

        # Control
        self.running = False
        self.autonomous_mode = False
        self.current_command = "keep forward"

        # Stats
        self.inference_count = 0
        self.control_count = 0
        self.last_waypoints = []

    def connect(self, host: str = 'localhost', port: int = 2000):
        """Connect to CARLA server"""
        print("Connecting to CARLA...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        version = self.client.get_server_version()
        print(f"  ✓ Connected to CARLA {version}")

        self.world = self.client.get_world()
        print(f"  ✓ Map: {self.world.get_map().name}")

    def spawn_vehicle(self, spawn_index: int = 0):
        """Spawn ego vehicle"""
        print("Spawning vehicle...")

        blueprint_library = self.world.get_blueprint_library()

        # Try different vehicle types
        vehicle_options = ['vehicle.tesla.model3', 'vehicle.dodge.charger', 'vehicle.audi.a2']
        vehicle_bp = None

        for option in vehicle_options:
            try:
                vehicles = blueprint_library.filter(option)
                if len(vehicles) > 0:
                    vehicle_bp = vehicles[0]
                    break
            except:
                continue

        if vehicle_bp is None:
            vehicle_bp = list(blueprint_library.filter('vehicle.*'))[0]

        print(f"  Using vehicle: {vehicle_bp.id}")

        # Set color
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,0,0')  # Red

        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available")

        # Try multiple spawn points if needed
        self.vehicle = None
        for i in range(len(spawn_points)):
            idx = (spawn_index + i) % len(spawn_points)
            spawn_point = spawn_points[idx]
            try:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle is not None:
                    print(f"  ✓ Vehicle spawned at spawn point {idx} (ID: {self.vehicle.id})")
                    break
            except Exception as e:
                print(f"  Spawn point {idx} failed: {e}")
                continue

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle at any spawn point")

        return self.vehicle

    def setup_cameras(self):
        """Setup 6 nuScenes-style cameras"""
        print("Setting up cameras...")

        self.camera_manager = CameraManager(
            self.world,
            self.vehicle,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT
        )
        self.camera_manager.spawn_cameras()
        print("  ✓ All 6 cameras ready")

    def setup_controller(self):
        """Setup vehicle controller"""
        print("Setting up controller...")

        self.state_tracker = VehicleStateTracker(self.vehicle)
        self.controller = WaypointVehicleController(
            lateral_controller=self.config.lateral_controller,
            target_speed=self.config.target_speed,
        )
        print("  ✓ Controller ready")

    def load_vla_model(self, use_mock: bool = True):
        """Load OpenDriveVLA model"""
        print("Loading VLA model...")

        if use_mock:
            print("  Using mock model for testing")
            self.vla_model = MockVLAModel()
        else:
            # Full model loading
            try:
                from carla_integration.opendrivevla_wrapper import OpenDriveVLAWrapper
                self.vla_model = OpenDriveVLAWrapper(
                    model_path=self.config.model_path,
                    use_bf16=self.config.use_bf16
                )
                self.vla_model.load_model()
            except Exception as e:
                print(f"  Warning: Failed to load VLA model: {e}")
                print("  Falling back to mock model")
                self.vla_model = MockVLAModel()

        print("  ✓ VLA model ready")

    def run_inference(self) -> List[Tuple[float, float]]:
        """Run VLA inference to get waypoints"""
        if self.vla_model is None:
            return []

        # Get ego state
        ego_state = self.state_tracker.get_ego_state()
        if ego_state is None:
            return []

        # Get historical trajectory
        history = self.state_tracker.get_historical_trajectory()

        # Run inference
        try:
            waypoints = self.vla_model.predict_trajectory(
                ego_state,
                history,
                self.current_command
            )
            self.inference_count += 1
            self.last_waypoints = waypoints
            return waypoints
        except Exception as e:
            print(f"Inference error: {e}")
            return []

    def apply_control(self, waypoints: List[Tuple[float, float]]):
        """Apply control based on waypoints"""
        if not waypoints:
            # No waypoints, apply brake
            control = carla.VehicleControl(throttle=0.0, brake=0.5)
            self.vehicle.apply_control(control)
            return

        # Get current speed
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2)

        # Compute control
        cmd = self.controller.compute_control(waypoints, speed)

        # Apply to CARLA
        control = carla.VehicleControl(
            throttle=cmd.throttle,
            steer=cmd.steer,
            brake=cmd.brake,
            hand_brake=cmd.hand_brake,
            reverse=cmd.reverse
        )
        self.vehicle.apply_control(control)
        self.control_count += 1

    def control_loop(self):
        """Main control loop"""
        print("\nStarting control loop...")
        print(f"  Inference rate: {self.config.inference_rate} Hz")
        print(f"  Target speed: {self.config.target_speed * 3.6:.1f} km/h")
        print("\nPress Ctrl+C to stop\n")

        inference_interval = 1.0 / self.config.inference_rate
        control_interval = 0.05  # 20 Hz control

        last_inference_time = 0.0
        last_control_time = 0.0
        last_print_time = 0.0

        self.running = True
        waypoints = []

        while self.running:
            current_time = time.time()

            # Update state tracker
            self.state_tracker.update()

            # Run inference at lower rate
            if self.autonomous_mode and current_time - last_inference_time >= inference_interval:
                waypoints = self.run_inference()
                last_inference_time = current_time

            # Apply control at higher rate
            if self.autonomous_mode and current_time - last_control_time >= control_interval:
                self.apply_control(waypoints)
                last_control_time = current_time

            # Print status
            if current_time - last_print_time >= 1.0:
                ego_state = self.state_tracker.get_ego_state()
                if ego_state:
                    speed_kmh = ego_state['speed'] * 3.6
                    print(f"Speed: {speed_kmh:5.1f} km/h | "
                          f"Inferences: {self.inference_count} | "
                          f"Controls: {self.control_count} | "
                          f"Waypoints: {len(waypoints)} | "
                          f"Mode: {'AUTO' if self.autonomous_mode else 'MANUAL'}")
                last_print_time = current_time

            # Small sleep to prevent CPU spinning
            time.sleep(0.01)

    def start_autonomous(self):
        """Enable autonomous mode"""
        self.autonomous_mode = True
        self.controller.reset()
        print("Autonomous mode ENABLED")

    def stop_autonomous(self):
        """Disable autonomous mode"""
        self.autonomous_mode = False
        # Apply brake
        control = carla.VehicleControl(throttle=0.0, brake=0.5)
        self.vehicle.apply_control(control)
        print("Autonomous mode DISABLED")

    def set_command(self, command: str):
        """Set navigation command"""
        valid_commands = ["keep forward", "turn left", "turn right"]
        if command in valid_commands:
            self.current_command = command
            print(f"Navigation command: {command}")

    def cleanup(self):
        """Clean up all resources"""
        print("\nCleaning up...")
        self.running = False

        # Stop vehicle
        if self.vehicle:
            try:
                control = carla.VehicleControl(throttle=0.0, brake=1.0)
                self.vehicle.apply_control(control)
            except:
                pass

        # Destroy cameras
        if self.camera_manager:
            self.camera_manager.destroy()

        # Destroy vehicle
        if self.vehicle:
            try:
                self.vehicle.destroy()
            except:
                pass

        print("✓ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="CARLA VLA Driver")
    parser.add_argument("--model-path", type=str,
                       default="checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct",
                       help="Path to VLA model checkpoint")
    parser.add_argument("--target-speed", type=float, default=5.0,
                       help="Target speed in m/s")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock model for testing")
    parser.add_argument("--host", type=str, default="localhost",
                       help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000,
                       help="CARLA server port")

    args = parser.parse_args()

    # Create config
    config = VLAConfig(
        model_path=args.model_path,
        target_speed=args.target_speed,
    )

    # Create driver
    driver = VLADriver(config)

    try:
        # Initialize
        driver.connect(args.host, args.port)
        driver.spawn_vehicle()
        driver.setup_cameras()
        driver.setup_controller()
        driver.load_vla_model(use_mock=args.mock)

        # Wait for cameras to start
        print("\nWaiting for camera data...")
        time.sleep(2.0)

        # Enable autonomous mode
        driver.start_autonomous()

        # Run control loop
        driver.control_loop()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.cleanup()


if __name__ == "__main__":
    main()
