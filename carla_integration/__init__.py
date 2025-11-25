"""
CARLA Integration Package for OpenDriveVLA

This package provides tools to integrate OpenDriveVLA model with CARLA simulator
for end-to-end autonomous driving.
"""

# Lazy imports to avoid torch dependency when not needed
__all__ = [
    'OpenDriveVLAWrapper',
    'EgoState',
    'HistoricalTrajectory',
    'MissionCommand',
    'WaypointToPIDController',
]

def __getattr__(name):
    """Lazy import to avoid loading torch when not needed"""
    if name in ('OpenDriveVLAWrapper', 'EgoState', 'HistoricalTrajectory',
                'MissionCommand', 'WaypointToPIDController'):
        from .opendrivevla_wrapper import (
            OpenDriveVLAWrapper,
            EgoState,
            HistoricalTrajectory,
            MissionCommand,
            WaypointToPIDController,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
