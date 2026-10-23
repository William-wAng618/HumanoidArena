"""Shared asynchronous episode recording helpers."""

from __future__ import annotations

import copy
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np


class AsyncEpisodeRecorder:
    """Background NPZ episode recorder with pluggable frame organizer."""

    def __init__(
        self,
        *,
        save_dir: str,
        task_name: str,
        organize_fn: Callable[[list[dict[str, Any]], int], dict[str, Any]],
        max_frames: int = 10000,
        max_save_workers: int = 1,
        max_queue_size: int = 10,
    ) -> None:
        self.save_dir = save_dir
        self.task_name = task_name
        self.organize_fn = organize_fn
        self.max_frames = max_frames
        self.max_save_workers = max(1, int(max_save_workers))

        os.makedirs(save_dir, exist_ok=True)

        self.is_recording = False
        self.recording_buffer: list[dict[str, Any]] = []
        self.frame_count = 0

        self.save_queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(max_queue_size)))
        self.thread_running = True
        self._lock = threading.Lock()
        self._active_workers = 0
        self.save_threads: list[threading.Thread] = []
        for worker_idx in range(self.max_save_workers):
            thread = threading.Thread(
                target=self._save_worker,
                name=f"episode-save-{worker_idx}",
                daemon=False,
            )
            thread.start()
            self.save_threads.append(thread)

    def _save_worker(self) -> None:
        while self.thread_running:
            try:
                task = self.save_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            data_buffer, timestamp_us, callback = task
            with self._lock:
                self._active_workers += 1
            try:
                success = self._save_to_disk(data_buffer, timestamp_us)
                if callback is not None:
                    callback(success)
            except Exception as exc:
                print(f"[AsyncEpisodeRecorder] save worker error: {exc}")
            finally:
                with self._lock:
                    self._active_workers = max(0, self._active_workers - 1)
                # Must run on every path: shutdown() joins the queue, so a missed
                # task_done() would hang the process at exit instead of just losing a frame.
                self.save_queue.task_done()

    def _save_to_disk(self, data_buffer: list[dict[str, Any]], timestamp_us: int) -> bool:
        temp_basename = f"{self.task_name}_{timestamp_us}_temp"
        filename = f"{self.task_name}_{timestamp_us}.npz"
        temp_filepath = os.path.join(self.save_dir, temp_basename)
        final_filepath = os.path.join(self.save_dir, filename)
        try:
            try:
                organized_data = self.organize_fn(
                    data_buffer,
                    timestamp_us,
                    self.save_dir,
                    self.task_name,
                )
            except TypeError:
                organized_data = self.organize_fn(data_buffer, timestamp_us)
            np.savez_compressed(temp_filepath, **organized_data)
            os.rename(temp_filepath + ".npz", final_filepath)
            print(
                f"[AsyncEpisodeRecorder] saved {len(data_buffer)} frames to {final_filepath}"
            )
            return True
        except Exception as exc:
            print(f"[AsyncEpisodeRecorder] save failed: {exc}")
            temp_npz = temp_filepath + ".npz"
            if os.path.exists(temp_npz):
                try:
                    os.remove(temp_npz)
                except OSError:
                    pass
            return False

    def start_recording(self) -> None:
        if self.is_recording:
            return
        self.is_recording = True
        self.recording_buffer = []
        self.frame_count = 0

    def add_frame(self, frame_data: dict[str, Any]) -> None:
        if not self.is_recording:
            return
        if self.frame_count >= self.max_frames:
            self.save_recording()
            return
        self.recording_buffer.append(copy.deepcopy(frame_data))
        self.frame_count += 1

    def save_recording(self, completion_callback: Callable[[bool], None] | None = None) -> None:
        if not self.is_recording:
            return
        if not self.recording_buffer:
            self.is_recording = False
            return
        self.is_recording = False
        timestamp_us = int(time.time() * 1_000_000)
        self.save_queue.put((self.recording_buffer, timestamp_us, completion_callback))
        self.recording_buffer = []
        self.frame_count = 0

    def cancel_recording(self) -> None:
        self.is_recording = False
        self.recording_buffer = []
        self.frame_count = 0

    def get_active_worker_count(self) -> int:
        with self._lock:
            return int(self._active_workers)

    def get_pending_save_count(self) -> int:
        return int(self.save_queue.qsize() + self.get_active_worker_count())

    def shutdown(self) -> None:
        if self.is_recording:
            self.cancel_recording()
        # Always drain, never gate on queue.empty(): a worker that has already dequeued a
        # pending save leaves the queue empty while the write is still in flight, so the
        # old `if not self.save_queue.empty()` guard skipped the wait and let the process
        # exit mid-write, leaving a truncated .npz (BadZipFile on read). _save_worker calls
        # task_done() on every path, so join() always returns once the write finishes.
        self.save_queue.join()
        self.thread_running = False
        for _ in self.save_threads:
            self.save_queue.put(None)
        for thread in self.save_threads:
            if thread.is_alive():
                thread.join(timeout=120.0)
