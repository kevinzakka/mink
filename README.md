# mink

A port of [pink](https://github.com/stephane-caron/pink) for MuJoCo.

## Installation

```bash
pip install -e ".[examples]"
```

## Todo list

- [ ] Improve collision avoidance collision pair creation.
- [ ] Add tests for collision avoidance limit.
- [ ] Implement `Jlog6` for frame task.
- [ ] Add tests for `mink.lie`.

## Usage

### Task costs

```python
from mink import FrameTask, PostureTask, ComTask

tasks = [
    pelvis_orientation_task := FrameTask(
        frame_name="pelvis",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=10.0,
    ),
    posture_task := PostureTask(cost=1e-1),
    com_task := ComTask(cost=200.0),
]

hand_tasks = []
for hand in hands:
    task = FrameTask(
        frame_name=hand,
        frame_type="site",
        position_cost=4.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    hand_tasks.append(task)
tasks.extend(hand_tasks)
```

### Task targets

```python
model = mujoco.MjModel.from_xml_path(_XML.as_posix())

configuration = mink.Configuration(model)

# Initialize from the "stand" keyframe configuration.
configuration.update_from_keyframe("stand")
pelvis_orientation_task.set_target_from_configuration(configuration)
posture_task.set_target_from_configuration(configuration)

# Set target from mocap (for interactive setting).
for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
    foot_task.set_target_from_mocap(data, feet_mocap_ids[i])
    hand_task.set_target_from_mocap(data, hands_mocap_ids[i])
```

### Limits

```python

# Joint position limits.
limits = [
    mink.ConfigurationLimit(model=model),
]
```

### Diff IK

```python
import time
from mink import solve_ik

solver = "quadprog"
dt = 0.02  # [s]
for t in np.arange(0.0, 5.0, dt):
    velocity = solve_ik(configuration, tasks, limits, dt, solver)
    configuration.integrate_inplace(velocity, dt)
    time.sleep(dt)
```

## Examples

```python
mjpython examples/arm_ur5e.py  # On macOS
python examples/arm_ur5e.py  # On Linux
```

## References

* https://hal.science/hal-04621130/file/OpenSoT_journal_wip.pdf
* https://hal.science/hal-04307572/document
* https://scaron.info/robotics/differential-inverse-kinematics.html
* https://www.roboticsproceedings.org/rss04/p20.pdf
