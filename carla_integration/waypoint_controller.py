"""
Waypoint-based Vehicle Controller for CARLA

This module converts waypoint predictions from OpenDriveVLA into
CARLA vehicle control commands (throttle, steer, brake).

Control Methods:
1. Pure Pursuit - Follow waypoints with lookahead distance
2. Stanley Controller - Heading + cross-track error control
3. MPC - Model Predictive Control (advanced)

Author: Claude Code
Date: 2025-11-25
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class VehicleState:
    """Current vehicle state from CARLA"""
    x: float           # World position X (meters)
    y: float           # World position Y (meters)
    z: float           # World position Z (meters)
    yaw: float         # Heading angle (radians, 0 = East, counter-clockwise positive)
    pitch: float       # Pitch angle (radians)
    roll: float        # Roll angle (radians)
    speed: float       # Forward speed (m/s)
    velocity_x: float  # World velocity X (m/s)
    velocity_y: float  # World velocity Y (m/s)
    velocity_z: float  # World velocity Z (m/s)
    acceleration_x: float  # Acceleration X (m/s^2)
    acceleration_y: float  # Acceleration Y (m/s^2)
    angular_velocity_z: float  # Yaw rate (rad/s)


@dataclass
class ControlCommand:
    """CARLA vehicle control command"""
    throttle: float  # [0, 1]
    steer: float     # [-1, 1], positive = left
    brake: float     # [0, 1]
    hand_brake: bool = False
    reverse: bool = False


class PurePursuitController:
    """
    Pure Pursuit Controller for trajectory following

    Follows waypoints by computing steering angle to reach a lookahead point
    on the trajectory. Simple and robust for most scenarios.

    Reference: https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf
    """

    def __init__(
        self,
        lookahead_distance: float = 3.0,
        min_lookahead: float = 2.0,
        max_lookahead: float = 6.0,
        lookahead_gain: float = 0.3,
        wheelbase: float = 2.875,  # Tesla Model 3 wheelbase
        max_steer: float = 0.7,    # Max steering angle (normalized)
    ):
        self.lookahead_distance = lookahead_distance
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.lookahead_gain = lookahead_gain
        self.wheelbase = wheelbase
        self.max_steer = max_steer

    def get_lookahead_distance(self, speed: float) -> float:
        """Compute speed-dependent lookahead distance"""
        ld = self.lookahead_distance + self.lookahead_gain * speed
        return np.clip(ld, self.min_lookahead, self.max_lookahead)

    def find_lookahead_point(
        self,
        waypoints: List[Tuple[float, float]],
        lookahead: float
    ) -> Tuple[float, float]:
        """
        Find the lookahead point on the trajectory

        Args:
            waypoints: List of (x, y) in ego coordinates
            lookahead: Lookahead distance in meters

        Returns:
            (x, y) of lookahead point in ego coordinates
        """
        if not waypoints:
            return (lookahead, 0.0)

        # Find point closest to lookahead distance
        for i, (x, y) in enumerate(waypoints):
            dist = np.sqrt(x**2 + y**2)
            if dist >= lookahead:
                return (x, y)

        # If no point is far enough, use the last point
        return waypoints[-1]

    def compute_steering(
        self,
        waypoints: List[Tuple[float, float]],
        speed: float
    ) -> float:
        """
        Compute steering angle using pure pursuit

        Args:
            waypoints: List of (x, y) in ego coordinates (+X forward, +Y left)
            speed: Current vehicle speed (m/s)

        Returns:
            Steering angle normalized to [-1, 1]
        """
        if not waypoints:
            return 0.0

        # Get lookahead point
        ld = self.get_lookahead_distance(speed)
        target_x, target_y = self.find_lookahead_point(waypoints, ld)

        # Distance to target
        target_dist = np.sqrt(target_x**2 + target_y**2)
        if target_dist < 0.5:
            return 0.0

        # Pure pursuit steering formula
        # delta = arctan(2 * L * sin(alpha) / ld)
        # where alpha is the angle to target, L is wheelbase
        alpha = np.arctan2(target_y, target_x)
        steering = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), target_dist)

        # Normalize to [-1, 1]
        steering = steering / np.deg2rad(70)  # Assuming 70 deg max steering
        steering = np.clip(steering, -self.max_steer, self.max_steer)

        return float(steering)


class StanleyController:
    """
    Stanley Controller for trajectory following

    Combines heading error and cross-track error for more precise control.
    Used by Stanford's winning entry in DARPA Grand Challenge.

    Reference: https://ieeexplore.ieee.org/document/4282788
    """

    def __init__(
        self,
        k_heading: float = 1.0,  # Heading error gain
        k_crosstrack: float = 2.5,  # Cross-track error gain
        k_soft: float = 1.0,  # Softening constant
        max_steer: float = 0.7,
    ):
        self.k_heading = k_heading
        self.k_crosstrack = k_crosstrack
        self.k_soft = k_soft
        self.max_steer = max_steer

    def compute_steering(
        self,
        waypoints: List[Tuple[float, float]],
        speed: float
    ) -> float:
        """
        Compute steering using Stanley controller

        Args:
            waypoints: List of (x, y) in ego coordinates
            speed: Current vehicle speed (m/s)

        Returns:
            Steering angle normalized to [-1, 1]
        """
        if not waypoints or len(waypoints) < 2:
            return 0.0

        # Cross-track error (lateral offset to nearest point on path)
        # In ego coords, this is simply the y-coordinate of the first waypoint
        crosstrack_error = waypoints[0][1]

        # Heading error (angle between vehicle heading and path direction)
        # Path direction from first to second waypoint
        dx = waypoints[1][0] - waypoints[0][0]
        dy = waypoints[1][1] - waypoints[0][1]
        path_heading = np.arctan2(dy, dx)  # Relative to ego, so this is the heading error

        # Stanley formula
        # steering = heading_error + arctan(k * crosstrack_error / (speed + k_soft))
        crosstrack_term = np.arctan2(
            self.k_crosstrack * crosstrack_error,
            speed + self.k_soft
        )
        steering = self.k_heading * path_heading + crosstrack_term

        # Normalize
        steering = steering / np.deg2rad(70)
        steering = np.clip(steering, -self.max_steer, self.max_steer)

        return float(steering)


class LongitudinalController:
    """
    PID controller for speed/throttle control

    Manages acceleration and braking to achieve target speed.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.05,
        max_throttle: float = 0.75,
        max_brake: float = 1.0,
        comfort_decel: float = 3.0,  # Comfortable deceleration (m/s^2)
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.comfort_decel = comfort_decel

        # PID state
        self._integral_error = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()

    def compute_control(
        self,
        current_speed: float,
        target_speed: float
    ) -> Tuple[float, float]:
        """
        Compute throttle and brake commands

        Args:
            current_speed: Current vehicle speed (m/s)
            target_speed: Desired speed (m/s)

        Returns:
            (throttle, brake) tuple
        """
        # Time delta
        current_time = time.time()
        dt = current_time - self._prev_time
        dt = max(dt, 0.01)  # Prevent division by zero
        self._prev_time = current_time

        # Speed error
        error = target_speed - current_speed

        # PID terms
        self._integral_error += error * dt
        self._integral_error = np.clip(self._integral_error, -10, 10)

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        # Control signal
        control = self.kp * error + self.ki * self._integral_error + self.kd * derivative

        # Split into throttle/brake
        if control > 0:
            throttle = np.clip(control, 0, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            # Smooth braking
            brake = np.clip(-control, 0, self.max_brake)

        return float(throttle), float(brake)

    def reset(self):
        """Reset PID state"""
        self._integral_error = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()


class WaypointVehicleController:
    """
    Complete vehicle controller combining lateral and longitudinal control

    Takes waypoints from OpenDriveVLA and outputs CARLA control commands.
    """

    def __init__(
        self,
        lateral_controller: str = "pure_pursuit",  # "pure_pursuit" or "stanley"
        target_speed: float = 5.0,
        min_speed: float = 0.5,
        max_speed: float = 15.0,
        speed_lookahead: int = 2,  # Which waypoint to use for speed estimation
        curvature_speed_factor: float = 0.5,  # Speed reduction for curves
    ):
        self.target_speed = target_speed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_lookahead = speed_lookahead
        self.curvature_speed_factor = curvature_speed_factor

        # Initialize controllers
        if lateral_controller == "pure_pursuit":
            self.lateral_controller = PurePursuitController()
        else:
            self.lateral_controller = StanleyController()

        self.longitudinal_controller = LongitudinalController()

        # State
        self._history = deque(maxlen=100)

    def estimate_curvature(self, waypoints: List[Tuple[float, float]]) -> float:
        """Estimate path curvature from waypoints"""
        if len(waypoints) < 3:
            return 0.0

        # Use 3 points to estimate curvature
        p1 = np.array(waypoints[0])
        p2 = np.array(waypoints[min(1, len(waypoints)-1)])
        p3 = np.array(waypoints[min(2, len(waypoints)-1)])

        # Triangle area method for curvature
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        if a < 0.01 or b < 0.01 or c < 0.01:
            return 0.0

        # Area using cross product
        area = abs(np.cross(p2 - p1, p3 - p1)) / 2.0

        # Curvature = 4 * area / (a * b * c)
        curvature = 4.0 * area / (a * b * c + 1e-6)
        return float(curvature)

    def compute_target_speed(
        self,
        waypoints: List[Tuple[float, float]],
        current_speed: float
    ) -> float:
        """
        Compute target speed based on waypoints and curvature

        Slows down for curves and speeds up on straight paths.
        """
        if not waypoints:
            return self.min_speed

        # Base target speed
        target = self.target_speed

        # Reduce speed based on curvature
        curvature = self.estimate_curvature(waypoints)
        curvature_factor = max(0.3, 1.0 - self.curvature_speed_factor * curvature * 10)
        target *= curvature_factor

        # Check if waypoints indicate stopping
        if len(waypoints) >= 2:
            # If trajectory is very short, slow down
            total_dist = sum(
                np.sqrt((waypoints[i+1][0] - waypoints[i][0])**2 +
                       (waypoints[i+1][1] - waypoints[i][1])**2)
                for i in range(len(waypoints)-1)
            )
            if total_dist < 5.0:  # Less than 5m trajectory
                target = min(target, total_dist / 2.0)

        return np.clip(target, self.min_speed, self.max_speed)

    def compute_control(
        self,
        waypoints: List[Tuple[float, float]],
        current_speed: float
    ) -> ControlCommand:
        """
        Compute full vehicle control from waypoints

        Args:
            waypoints: List of (x, y) in ego coordinates (+X forward, +Y left)
            current_speed: Current vehicle speed (m/s)

        Returns:
            ControlCommand with throttle, steer, brake
        """
        # Handle empty waypoints
        if not waypoints:
            return ControlCommand(throttle=0.0, steer=0.0, brake=0.5)

        # Compute steering
        steer = self.lateral_controller.compute_steering(waypoints, current_speed)

        # Compute target speed
        target_speed = self.compute_target_speed(waypoints, current_speed)

        # Compute throttle/brake
        throttle, brake = self.longitudinal_controller.compute_control(
            current_speed, target_speed
        )

        # Safety: increase brake when steering is high
        if abs(steer) > 0.5 and current_speed > 5.0:
            brake = max(brake, 0.3)
            throttle *= 0.5

        return ControlCommand(
            throttle=throttle,
            steer=steer,
            brake=brake
        )

    def reset(self):
        """Reset controller state"""
        self.longitudinal_controller.reset()
        self._history.clear()


class HistoryTracker:
    """
    Tracks vehicle history for OpenDriveVLA input

    Maintains historical trajectory points needed for model inference.
    """

    def __init__(self, history_seconds: float = 2.0, sample_rate: float = 2.0):
        """
        Args:
            history_seconds: How many seconds of history to keep
            sample_rate: Samples per second
        """
        self.history_seconds = history_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(history_seconds * sample_rate)

        self._positions = deque(maxlen=self.max_samples + 10)
        self._timestamps = deque(maxlen=self.max_samples + 10)
        self._last_sample_time = 0.0

    def update(self, x: float, y: float, timestamp: Optional[float] = None):
        """
        Update with new position

        Args:
            x, y: Position in world coordinates
            timestamp: Time in seconds (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()

        # Only sample at desired rate
        if timestamp - self._last_sample_time >= 1.0 / self.sample_rate:
            self._positions.append((x, y))
            self._timestamps.append(timestamp)
            self._last_sample_time = timestamp

    def get_history_in_ego_frame(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float
    ) -> List[Tuple[float, float]]:
        """
        Get historical trajectory in ego vehicle frame

        Args:
            current_x, current_y: Current position in world coordinates
            current_yaw: Current heading in radians

        Returns:
            List of 4 points at [-2.0s, -1.5s, -1.0s, -0.5s] in ego coordinates
        """
        if len(self._positions) < 4:
            # Not enough history, return zeros
            return [(0.0, 0.0)] * 4

        # Get positions at desired timestamps
        current_time = self._timestamps[-1] if self._timestamps else time.time()
        target_times = [
            current_time - 2.0,
            current_time - 1.5,
            current_time - 1.0,
            current_time - 0.5
        ]

        result = []
        for target_time in target_times:
            # Find closest sample
            pos = self._interpolate_position(target_time)

            # Transform to ego frame
            dx = pos[0] - current_x
            dy = pos[1] - current_y

            # Rotate to ego frame
            cos_yaw = np.cos(-current_yaw)
            sin_yaw = np.sin(-current_yaw)
            ego_x = dx * cos_yaw - dy * sin_yaw
            ego_y = dx * sin_yaw + dy * cos_yaw

            result.append((ego_x, ego_y))

        return result

    def _interpolate_position(self, target_time: float) -> Tuple[float, float]:
        """Interpolate position at target time"""
        if not self._timestamps:
            return (0.0, 0.0)

        # Find bracketing samples
        for i in range(len(self._timestamps) - 1):
            t0, t1 = self._timestamps[i], self._timestamps[i + 1]
            if t0 <= target_time <= t1:
                # Linear interpolation
                alpha = (target_time - t0) / (t1 - t0 + 1e-6)
                x = self._positions[i][0] * (1 - alpha) + self._positions[i + 1][0] * alpha
                y = self._positions[i][1] * (1 - alpha) + self._positions[i + 1][1] * alpha
                return (x, y)

        # Return closest edge
        if target_time < self._timestamps[0]:
            return self._positions[0]
        return self._positions[-1]

    def clear(self):
        """Clear history"""
        self._positions.clear()
        self._timestamps.clear()


def main():
    """Test the controllers"""
    print("=" * 60)
    print("Waypoint Controller Test")
    print("=" * 60)

    # Test waypoints (forward curve)
    waypoints = [
        (2.0, 0.5),
        (4.0, 1.2),
        (6.0, 2.0),
        (8.0, 2.5),
        (10.0, 2.8),
        (12.0, 3.0)
    ]

    # Test pure pursuit
    print("\n1. Pure Pursuit Controller:")
    pp = PurePursuitController()
    steer = pp.compute_steering(waypoints, 5.0)
    print(f"   Steering: {steer:.3f}")

    # Test stanley
    print("\n2. Stanley Controller:")
    stanley = StanleyController()
    steer = stanley.compute_steering(waypoints, 5.0)
    print(f"   Steering: {steer:.3f}")

    # Test full controller
    print("\n3. Full Vehicle Controller:")
    controller = WaypointVehicleController(target_speed=8.0)
    cmd = controller.compute_control(waypoints, 5.0)
    print(f"   Throttle: {cmd.throttle:.3f}")
    print(f"   Steer: {cmd.steer:.3f}")
    print(f"   Brake: {cmd.brake:.3f}")

    # Test history tracker
    print("\n4. History Tracker:")
    tracker = HistoryTracker()

    # Simulate driving forward
    for i in range(20):
        x = i * 0.5
        y = 0.0
        tracker.update(x, y, i * 0.1)

    history = tracker.get_history_in_ego_frame(10.0, 0.0, 0.0)
    print(f"   Historical points (ego frame): {history}")


if __name__ == "__main__":
    main()
