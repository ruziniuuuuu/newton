# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Cloth Franka
#
# This simulation demonstrates a coupled robot-cloth simulation
# using the VBD solver for the cloth and Featherstone for the robot,
# showcasing its ability to handle complex contacts while ensuring it
# remains intersection-free.
#
# Command: python -m newton.examples cloth_franka
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD
from newton.utils import transform_twist


@wp.kernel
def compute_ee_delta(
    body_q: wp.array(dtype=wp.transform),
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    # compute pose difference between end effector and target
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


def compute_body_jacobian(
    model: Model,
    joint_q: wp.array,
    joint_qd: wp.array,
    body_id: int | str,  # Can be either body index or body name
    offset: wp.transform | None = None,
    velocity: bool = True,
    include_rotation: bool = False,
):
    if isinstance(body_id, str):
        body_id = model.body_name.get(body_id)
    if offset is None:
        offset = wp.transform_identity()

    joint_q.requires_grad = True
    joint_qd.requires_grad = True

    if velocity:

        @wp.kernel
        def compute_body_out(body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)):
            # TODO verify transform twist
            mv = transform_twist(offset, body_qd[body_id])
            if wp.static(include_rotation):
                for i in range(6):
                    body_out[i] = mv[i]
            else:
                for i in range(3):
                    body_out[i] = mv[3 + i]

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3
    else:

        @wp.kernel
        def compute_body_out(body_q: wp.array(dtype=wp.transform), body_out: wp.array(dtype=float)):
            tf = body_q[body_id] * offset
            if wp.static(include_rotation):
                for i in range(7):
                    body_out[i] = tf[i]
            else:
                for i in range(3):
                    body_out[i] = tf[i]

        in_dim = model.joint_coord_count
        out_dim = 7 if include_rotation else 3

    out_state = model.state(requires_grad=True)
    body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
        eval_fk(model, joint_q, joint_qd, out_state)
        wp.launch(compute_body_out, 1, inputs=[out_state.body_qd if velocity else out_state.body_q], outputs=[body_out])

    def onehot(i):
        x = np.zeros(out_dim, dtype=np.float32)
        x[i] = 1.0
        return wp.array(x)

    J = np.empty((out_dim, in_dim), dtype=wp.float32)
    for i in range(out_dim):
        tape.backward(grads={body_out: onehot(i)})
        J[i] = joint_qd.grad.numpy() if velocity else joint_q.grad.numpy()
        tape.zero()
    return J.astype(np.float32)


class Example:
    def __init__(self, viewer):
        # parameters
        #   simulation
        self.add_cloth = True
        self.add_robot = True
        self.sim_substeps = 15
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        #   contact
        #       body-cloth contact
        self.cloth_particle_radius = 0.008
        self.cloth_body_contact_margin = 0.01
        #       self-contact
        self.self_contact_radius = 0.002
        self.self_contact_margin = 0.003

        self.soft_contact_ke = 100
        self.soft_contact_kd = 2e-3

        self.robot_friction = 1.0
        self.table_friction = 0.5
        self.self_contact_friction = 0.25

        #   elasticity
        self.tri_ke = 1e2
        self.tri_ka = 1e2
        self.tri_kd = 1.5e-6

        self.bending_ke = 1e-4
        self.bending_kd = 1e-3

        self.scene = ModelBuilder()
        self.soft_contact_max = 1000000

        self.viewer = viewer
        self._renderer_is_key_down_raw = None

        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)

            self.scene.add_builder(franka)
            self.bodies_per_world = franka.body_count
            self.dof_q_per_world = franka.joint_coord_count
            self.dof_qd_per_world = franka.joint_dof_count

        # add a table
        self.scene.add_shape_box(
            -1,
            wp.transform(
                wp.vec3(0.0, -0.5, 0.1),
                wp.quat_identity(),
            ),
            hx=0.4,
            hy=0.4,
            hz=0.1,
        )

        # add the T-shirt
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/shirt"))
        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        vertices = [wp.vec3(v) for v in mesh_points]

        if self.add_cloth:
            self.scene.add_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
                pos=wp.vec3(0.0, 0.70, 0.28),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=0.2,
                scale=0.01,
                tri_ke=self.tri_ke,
                tri_ka=self.tri_ka,
                tri_kd=self.tri_kd,
                edge_ke=self.bending_ke,
                edge_kd=self.bending_kd,
                particle_radius=self.cloth_particle_radius,
            )

            self.scene.color()

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.sim_time = 0.0

        # initialize robot solver
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.cloth_solver: SolverVBD | None = None
        if self.add_cloth:
            # initialize cloth solver
            #   set edge rest angle to zero to disable bending, this is currently a workaround to make SolverVBD stable
            #   TODO: fix SolverVBD's bending issue
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=self.iterations,
                self_contact_radius=self.self_contact_radius,
                self_contact_margin=self.self_contact_margin,
                handle_self_contact=True,
                vertex_collision_buffer_pre_alloc=32,
                edge_collision_buffer_pre_alloc=64,
                integrate_with_external_rigid_solver=True,
                collision_detection_interval=-1,
            )

        self.viewer.set_model(self.model)
        self._configure_viewer_for_teleop()

        # create Warp arrays for gravity so we can swap Model.gravity during
        # a simulation running under CUDA graph capture
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)  # used for the robot solver
        # gravity in cm/s^2
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)  # used for the cloth solver

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # graph capture
        if self.add_cloth:
            self.capture()

    def set_up_control(self):
        self.control = self.model.control()

        # we are controlling the velocity
        out_dim = 6
        in_dim = self.model.joint_dof_count

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        # for robot control
        self.delta_q = wp.empty(self.model.joint_count, dtype=float)
        self.joint_q_des = wp.array(self.model.joint_q.numpy(), dtype=float)

        @wp.kernel
        def compute_body_out(body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)):
            # TODO verify transform twist
            mv = transform_twist(wp.static(self.endeffector_offset), body_qd[wp.static(self.endeffector_id)])
            for i in range(6):
                body_out[i] = mv[i]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)

        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)

        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.initial_pose = self.model.joint_q.numpy()

        # Teleoperation parameters
        self.teleop_linear_speed = 0.25  # meters per second
        self.teleop_angular_speed = 1.2  # radians per second
        self.teleop_gripper_speed = 2.0  # normalized units per second
        self.teleop_fast_scale = 2.0
        self.teleop_slow_scale = 0.25
        self.finger_open_limit = 0.04

        finger_average = float(np.mean(self.initial_pose[-2:])) if self.initial_pose.size >= 2 else 0.0
        if self.finger_open_limit > 0.0:
            self.gripper_open_ratio = float(np.clip(finger_average / self.finger_open_limit, 0.0, 1.0))
        else:
            self.gripper_open_ratio = 0.0
        self.teleop_help_printed = False
        self.last_active_keys: list[str] = []

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _configure_viewer_for_teleop(self):
        """
        Disable default camera WASD controls so keyboard input can fully drive teleoperation.
        """
        viewer = self.viewer
        if viewer is None:
            return

        renderer = getattr(viewer, "renderer", None)
        if renderer is None or not hasattr(renderer, "is_key_down"):
            return

        if self._renderer_is_key_down_raw is not None:
            return

        try:
            import pyglet  # noqa: PLC0415
        except Exception:
            return

        blocked_keys = {
            pyglet.window.key.W,
            pyglet.window.key.A,
            pyglet.window.key.S,
            pyglet.window.key.D,
        }
        original_is_key_down = renderer.is_key_down
        self._renderer_is_key_down_raw = original_is_key_down

        def filtered_is_key_down(key_code, *, _original=original_is_key_down, _blocked=blocked_keys):
            if key_code in _blocked:
                return False
            return _original(key_code)

        renderer.is_key_down = filtered_is_key_down  # type: ignore[assignment]

        print("Teleop: viewer camera now ignores W/A/S/D so the robot can use them.")

    def _teleop_is_key_down(self, key: str) -> bool:
        try:
            import pyglet  # noqa: PLC0415
        except Exception:
            return False

        key = key.lower()

        if len(key) == 1 and key.isalpha():
            key_code = getattr(pyglet.window.key, key.upper(), None)
        elif len(key) == 1 and key.isdigit():
            key_code = getattr(pyglet.window.key, f"_{key}", None)
        else:
            special_keys = {
                "space": pyglet.window.key.SPACE,
                "escape": pyglet.window.key.ESCAPE,
                "esc": pyglet.window.key.ESCAPE,
                "enter": pyglet.window.key.ENTER,
                "return": pyglet.window.key.ENTER,
                "tab": pyglet.window.key.TAB,
                "shift": pyglet.window.key.LSHIFT,
                "ctrl": pyglet.window.key.LCTRL,
                "alt": pyglet.window.key.LALT,
                "up": pyglet.window.key.UP,
                "down": pyglet.window.key.DOWN,
                "left": pyglet.window.key.LEFT,
                "right": pyglet.window.key.RIGHT,
                "backspace": pyglet.window.key.BACKSPACE,
                "delete": pyglet.window.key.DELETE,
            }
            key_code = special_keys.get(key, None)

        if key_code is None:
            return False

        if self._renderer_is_key_down_raw is not None:
            return self._renderer_is_key_down_raw(key_code)

        if self.viewer is None:
            return False

        return self.viewer.is_key_down(key)

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                (-0.5, -0.5, -0.1),
                # (-0.5, -0.2, 0.5),
                wp.quat_identity(),
            ),
            floating=False,
            scale=1,  # unit: cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        clamp_close_activation_val = 0.06
        clamp_open_activation_val = 0.8

        self.robot_key_poses = np.array(
            [
                # translation_duration, gripper transform (3D position, 4D quaternion), gripper open (1) or closed (0)
                # # top left
                [2.5, 0.31, -0.60, 0.23, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.31, -0.60, 0.23, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.26, -0.60, 0.26, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.12, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, -0.06, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, -0.06, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # bottom right
                [2, 0.15, -0.33, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3, 0.15, -0.33, 0.21, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3, 0.15, -0.33, 0.21, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.15, -0.33, 0.28, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, -0.02, -0.33, 0.28, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, -0.02, -0.33, 0.28, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # top left
                [2, -0.28, -0.60, 0.28, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -0.28, -0.60, 0.20, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -0.28, -0.60, 0.20, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -0.18, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, 0.05, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, 0.05, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # # bottom left
                [3, -0.18, -0.30, 0.205, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3, -0.18, -0.30, 0.205, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -0.03, -0.30, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, -0.03, -0.30, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -0.03, -0.30, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # bottom
                [2, -0.0, -0.21, 0.30, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -0.0, -0.21, 0.20, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -0.0, -0.21, 0.20, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -0.0, -0.21, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, -0.0, -0.30, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1.5, -0.0, -0.30, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1.5, -0.0, -0.40, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, -0.0, -0.40, 0.35, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
            ],
            dtype=np.float32,
        )
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform(
            [
                0.0,
                0.0,
                0.22,
            ],
            wp.quat_identity(),
        )

    def compute_body_jacobian(
        self,
        model: Model,
        joint_q: wp.array,
        joint_qd: wp.array,
        include_rotation: bool = False,
    ):
        """
        Compute the Jacobian of the end effector's velocity related to joint_q

        """

        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel, 1, inputs=[self.temp_state_for_jacobian.body_qd], outputs=[self.body_out]
            )

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def franka_teleop(self) -> np.ndarray:
        """
        Map keyboard input to a 6D end-effector twist command (vx, vy, vz, wx, wy, wz).

        Returns:
            np.ndarray: Desired end-effector velocities represented as [vx, vy, vz, wx, wy, wz].
        """
        command = np.zeros(6, dtype=np.float32)

        if not self.teleop_help_printed:
            print(
                "Teleop controls: W/S=X, A/D=Y, R/F=Z, U/O=Roll, I/K=Pitch, J/L=Yaw, Q=open gripper, E=close, "
                "SHIFT=fast, CTRL=slow."
            )
            self.teleop_help_printed = True

        key_down = self._teleop_is_key_down

        linear_dir = np.zeros(3, dtype=np.float32)
        angular_dir = np.zeros(3, dtype=np.float32)
        active_keys: list[str] = []

        # Linear motion (W/S for X, A/D for Y, R/F for Z)
        if key_down("w"):
            linear_dir[0] += 1.0
            active_keys.append("w")
        if key_down("s"):
            linear_dir[0] -= 1.0
            active_keys.append("s")
        if key_down("a"):
            linear_dir[1] += 1.0
            active_keys.append("a")
        if key_down("d"):
            linear_dir[1] -= 1.0
            active_keys.append("d")
        if key_down("r"):
            linear_dir[2] += 1.0
            active_keys.append("r")
        if key_down("f"):
            linear_dir[2] -= 1.0
            active_keys.append("f")

        # Angular motion (U/O roll, I/K pitch, J/L yaw)
        if key_down("u"):
            angular_dir[0] += 1.0
            active_keys.append("u")
        if key_down("o"):
            angular_dir[0] -= 1.0
            active_keys.append("o")
        if key_down("i"):
            angular_dir[1] += 1.0
            active_keys.append("i")
        if key_down("k"):
            angular_dir[1] -= 1.0
            active_keys.append("k")
        if key_down("j"):
            angular_dir[2] += 1.0
            active_keys.append("j")
        if key_down("l"):
            angular_dir[2] -= 1.0
            active_keys.append("l")

        speed_scale = 1.0
        if key_down("shift"):
            speed_scale *= self.teleop_fast_scale
            active_keys.append("shift")
        if key_down("ctrl"):
            speed_scale *= self.teleop_slow_scale
            active_keys.append("ctrl")

        # Normalize diagonal input to keep speed consistent
        linear_norm = np.linalg.norm(linear_dir)
        if linear_norm > 1.0:
            linear_dir /= linear_norm
        angular_norm = np.linalg.norm(angular_dir)
        if angular_norm > 1.0:
            angular_dir /= angular_norm

        command[:3] = linear_dir * self.teleop_linear_speed * speed_scale
        command[3:] = angular_dir * self.teleop_angular_speed * speed_scale

        # Gripper (Z to open, X to close)
        if key_down("z"):
            self.gripper_open_ratio = min(1.0, self.gripper_open_ratio + self.teleop_gripper_speed * self.frame_dt)
            active_keys.append("z")
        if key_down("x"):
            self.gripper_open_ratio = max(0.0, self.gripper_open_ratio - self.teleop_gripper_speed * self.frame_dt)
            active_keys.append("x")

        # Emit a terminal prompt whenever the input changes to avoid spamming the terminal
        if active_keys != self.last_active_keys:
            command_preview = ", ".join(f"{c:.3f}" for c in command)
            gripper_pct = self.gripper_open_ratio * 100.0
            keys_preview = " ".join(active_keys) if active_keys else "none"
            print(f"Teleop keys: {keys_preview} | command: [{command_preview}] | gripper: {gripper_pct:.1f}%")
            self.last_active_keys = active_keys.copy()

        return command

    def generate_control_joint_qd(
        self,
        state_in: State,
    ):
        include_rotation = True
        delta_target = self.franka_teleop()

        self.compute_body_jacobian(
            self.model,
            state_in.joint_q,
            state_in.joint_qd,
            include_rotation=include_rotation,
        )
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        J_inv = np.linalg.pinv(J)

        # 2. Compute null-space projector
        #    I is size [num_joints x num_joints]
        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = state_in.joint_q.numpy()

        # 3. Define a desired "elbow-up" reference posture
        #    (For example, one that keeps joint 2 or 3 above a certain angle.)
        #    Adjust indices and angles to your robot's kinematics.
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]  # e.g., set elbow joint around 1 rad to keep it up

        # 4. Define a null-space velocity term pulling joints toward q_des
        #    K_null is a small gain so it doesn't override main task
        K_null = 1.0
        delta_q_null = K_null * (q_des - q)

        # 5. Combine primary task and null-space controller
        delta_q = J_inv @ delta_target + N @ delta_q_null

        # Apply gripper finger control
        finger_target = self.gripper_open_ratio * self.finger_open_limit
        delta_q[-2] = finger_target - q[-2]
        delta_q[-1] = finger_target - q[-1]

        self.target_joint_qd.assign(delta_q.astype(np.float32))

    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            # robot sim
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                # set particle_count = 0 to disable particle simulation in robot solver
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                # Update the robot pose - this will modify state_0 and copy to state_1
                self.model.shape_contact_pair_count = 0

                self.state_0.joint_qd.assign(self.target_joint_qd)
                # Just update the forward kinematics to get body positions from joint coordinates
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                self.state_0.particle_f.zero_()

                # restore original settings
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # cloth sim
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.cloth_body_contact_margin)

            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test(self):
        p_lower = wp.vec3(-0.34, -0.9, 0.0)
        p_upper = wp.vec3(0.34, 0.0, 0.51)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 2.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 0.7,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=3850)
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer)

    newton.examples.run(example, args)
