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
# Example Galbot Golf
#
# Demonstrates loading the Galbot One Golf robot articulation directly from
# an MJCF file using newton.ModelBuilder.add_mjcf().
#
# Command: python -m newton.examples galbot_golf --num-worlds 4
#
###########################################################################

from pathlib import Path

import warp as wp

import newton
import newton.examples

import os

class Example:
    def __init__(self, viewer, num_worlds: int = 4):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds
        self.viewer = viewer

        sn_assets_folder = os.getenv("SYNTHNOVA_ASSETS")
        mjcf_path = sn_assets_folder + "/synthnova_assets/robot/galbot_one_golf/mjcf/galbot_one_golf.xml"

        galbot = newton.ModelBuilder()
        galbot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1.0e-5)
        galbot.default_shape_cfg.ke = 5.0e4
        galbot.default_shape_cfg.kd = 5.0e2
        galbot.default_shape_cfg.kf = 1.0e3
        galbot.default_shape_cfg.mu = 0.75

        galbot.add_mjcf(
            str(mjcf_path),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)),
            floating=True,
            # ignore_classes=("visual"),
            collapse_fixed_joints=False,
            parse_visuals_as_colliders=False,
            enable_self_collisions=False,
        )

        # for dof_index in range(len(galbot.joint_dof_mode)):
        #     galbot.joint_dof_mode[dof_index] = newton.JointMode.TARGET_POSITION
        #     galbot.joint_target_ke[dof_index] = 800.0
        #     galbot.joint_target_kd[dof_index] = 40.0

        # galbot.approximate_meshes("bounding_box")

        builder = newton.ModelBuilder()
        builder.replicate(galbot, self.num_worlds, spacing=(4.0, 4.0, 0.0))
        builder.add_ground_plane()

        self.model = builder.finalize()

        initial_target = wp.clone(self.model.joint_q)
        self.model.joint_target = wp.clone(initial_target)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            njmax=400,
            ncon_per_world=256,  # ensure enough contact slots for dense collisions
            cone="elliptic",
            impratio=100,
            iterations=100,
            ls_iterations=50,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc.
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "Galbot bodies stay above ground",
            lambda q, qd: q[2] > 0.0,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=1, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds)

    newton.examples.run(example, args)
