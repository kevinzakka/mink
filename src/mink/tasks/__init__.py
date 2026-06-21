"""Kinematic tasks."""

from .com_task import ComTask as ComTask
from .damping_task import DampingTask as DampingTask
from .dof_freezing_task import DofFreezingTask as DofFreezingTask
from .equality_constraint_task import EqualityConstraintTask as EqualityConstraintTask
from .frame_task import FrameTask as FrameTask
from .kinetic_energy_regularization_task import (
    KineticEnergyRegularizationTask as KineticEnergyRegularizationTask,
)
from .look_at_task import LookAtTask as LookAtTask
from .posture_task import PostureTask as PostureTask
from .relative_frame_task import RelativeFrameTask as RelativeFrameTask
from .task import BaseTask as BaseTask
from .task import Objective as Objective
from .task import Task as Task
