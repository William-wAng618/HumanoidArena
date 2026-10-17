from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rerecord_sonic_recordings_to_multicam as sonic_rerecord
import rerecord_twist2_recordings_to_multicam as twist2_rerecord
from rerecord_parallel_utils import build_worker_runtime_config


class RerecordInputFilterTests(unittest.TestCase):
    def test_sonic_build_jobs_skips_temp_and_bad_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_root = tmp_path / "sonic"
            source_root.mkdir()

            good_file = source_root / "ok.npz"
            np.savez_compressed(
                good_file,
                schema_version=np.array("sonic_episode_v3"),
                task=np.array("Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody"),
            )
            (source_root / "broken.npz").write_bytes(b"not-a-zip")
            (source_root / "leftover_temp.npz").write_bytes(b"partial")

            original_mapping = dict(sonic_rerecord.TASK_TO_ENV_CONFIG)
            try:
                sonic_rerecord.TASK_TO_ENV_CONFIG["Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody"] = (
                    tmp_path / "dummy_env.yaml"
                )
                args = SimpleNamespace(
                    source_roots=[str(source_root)],
                    output_suffix="_multicam_rerecord",
                    force=True,
                )
                jobs = sonic_rerecord.build_jobs(args)
            finally:
                sonic_rerecord.TASK_TO_ENV_CONFIG.clear()
                sonic_rerecord.TASK_TO_ENV_CONFIG.update(original_mapping)

            self.assertEqual(len(jobs), 1)
            self.assertEqual(Path(jobs[0]["source_file"]).resolve(), good_file.resolve())

    def test_sonic_build_command_enables_perspective_camera(self) -> None:
        args = SimpleNamespace(
            device="cpu",
            robot_type="g129",
            replay_mode="direct_replay",
            task_runtime_profile="auto",
            sonic_encoder_path="/tmp/encoder.onnx",
            sonic_decoder_path="/tmp/decoder.onnx",
            recording_save_workers=1,
            recording_save_queue_size=4,
            seed=42,
            disable_cameras=False,
            enable_perspective_camera=True,
            disable_front_camera=False,
            disable_wrist_cameras=False,
            disable_dex3=True,
            headless=True,
        )
        job = {
            "source_file": Path("/tmp/source.npz"),
            "env_config": Path("/tmp/env.yaml"),
            "task_name": "Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody",
        }
        runtime_config = build_worker_runtime_config(
            worker_index=0,
            image_port_base=5600,
            image_port_stride=10,
            process_id=4321,
            session_token="deadbeef",
        )

        command = sonic_rerecord.build_command(
            args,
            job,
            Path("/tmp/output"),
            "/usr/bin/python",
            runtime_config,
        )

        self.assertIn("--enable_cameras", command)
        self.assertIn("--task_runtime_profile", command)
        self.assertIn("auto", command)
        self.assertIn("--enable_wrist_cameras", command)
        self.assertIn("--enable_perspective_camera", command)
        self.assertIn("--world_camera_port", command)
        self.assertIn("5601", command)

    def test_sonic_build_command_can_disable_front_and_wrist_cameras(self) -> None:
        args = SimpleNamespace(
            device="cpu",
            robot_type="g129",
            replay_mode="direct_replay",
            task_runtime_profile="replay_compat",
            sonic_encoder_path="/tmp/encoder.onnx",
            sonic_decoder_path="/tmp/decoder.onnx",
            recording_save_workers=1,
            recording_save_queue_size=4,
            seed=42,
            disable_cameras=False,
            enable_perspective_camera=True,
            disable_front_camera=True,
            disable_wrist_cameras=True,
            disable_dex3=True,
            headless=True,
        )
        job = {
            "source_file": Path("/tmp/source.npz"),
            "env_config": Path("/tmp/env.yaml"),
            "task_name": "Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody",
        }
        runtime_config = build_worker_runtime_config(
            worker_index=0,
            image_port_base=5600,
            image_port_stride=10,
            process_id=4321,
            session_token="deadbeef",
        )

        command = sonic_rerecord.build_command(
            args,
            job,
            Path("/tmp/output"),
            "/usr/bin/python",
            runtime_config,
        )

        self.assertIn("--enable_cameras", command)
        self.assertIn("--task_runtime_profile", command)
        self.assertIn("replay_compat", command)
        self.assertIn("--enable_perspective_camera", command)
        self.assertIn("--disable_front_camera", command)
        self.assertIn("--disable_wrist_cameras", command)
        self.assertNotIn("--enable_wrist_cameras", command)

    def test_twist2_discovery_skips_temp_and_bad_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_root = tmp_path / "twist2"
            output_root = tmp_path / "twist2_multicam_rerecord"
            input_root.mkdir()
            output_root.mkdir()

            good_file = input_root / "ok.npz"
            np.savez_compressed(
                good_file,
                task=np.array("Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody"),
            )
            (input_root / "broken.npz").write_bytes(b"not-a-zip")
            (input_root / "leftover_temp.npz").write_bytes(b"partial")

            npz_paths = twist2_rerecord.find_npz_files(input_root)
            args = SimpleNamespace(env_config_yaml="")
            jobs = twist2_rerecord.build_jobs(args, input_root, output_root, npz_paths)

            self.assertEqual(len(jobs), 1)
            self.assertEqual(Path(jobs[0]["source_file"]).resolve(), good_file.resolve())
            self.assertTrue(str(jobs[0]["env_config_yaml"]).endswith("pickplace_box_twist2.yaml"))

    def test_twist2_build_jobs_uses_task_env_config_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_root = tmp_path / "twist2"
            output_root = tmp_path / "twist2_multicam_rerecord"
            input_root.mkdir()
            output_root.mkdir()

            football_file = input_root / "football.npz"
            np.savez_compressed(
                football_file,
                task=np.array("Isaac-Move-Football-Single-G129-Dex3-Wholebody"),
            )

            args = SimpleNamespace(env_config_yaml="")
            jobs = twist2_rerecord.build_jobs(args, input_root, output_root, [football_file])

            self.assertEqual(len(jobs), 1)
            self.assertTrue(str(jobs[0]["env_config_yaml"]).endswith("football_single_twist2.yaml"))

    def test_twist2_build_command_can_record_perspective_only(self) -> None:
        args = SimpleNamespace(
            device="cpu",
            env_config_yaml="/tmp/env.yaml",
            robot_type="g129",
            replay_mode="direct",
            task_runtime_profile="auto",
            recording_save_workers=1,
            recording_save_queue_size=4,
            seed=42,
            enable_perspective_camera=True,
            disable_front_camera=True,
            disable_wrist_cameras=True,
            headless=True,
        )
        runtime_config = build_worker_runtime_config(
            worker_index=0,
            image_port_base=5600,
            image_port_stride=10,
            process_id=4321,
            session_token="deadbeef",
        )

        command = twist2_rerecord.build_command(
            args,
            python_bin="/usr/bin/python",
            replay_file=Path("/tmp/source.npz"),
            output_dir=Path("/tmp/output"),
            task_name="Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody",
            env_config_yaml="/tmp/env.yaml",
            runtime_config=runtime_config,
        )

        self.assertIn("--enable_cameras", command)
        self.assertIn("--task_runtime_profile", command)
        self.assertIn("auto", command)
        self.assertIn("--enable_perspective_camera", command)
        self.assertIn("--disable_front_camera", command)
        self.assertIn("--disable_wrist_cameras", command)
        self.assertNotIn("--enable_wrist_cameras", command)
        self.assertIn("--world_camera_port", command)
        self.assertIn("5601", command)


if __name__ == "__main__":
    unittest.main()
