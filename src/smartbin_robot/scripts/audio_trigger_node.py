#!/usr/bin/env python3
import json
import queue
import sys
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd
import vosk

WAKE_PHRASE = "robot come here"
DISMISS_PHRASE = "robot go home"

# Audeze Maxwell (USB) only exposes 48kHz. Vosk expects 16kHz.
# Capture at 48kHz and downsample by factor 3 before feeding Vosk.
CAPTURE_RATE = 48000
VOSK_RATE = 16000
DOWNSAMPLE = CAPTURE_RATE // VOSK_RATE # = 3
BLOCKSIZE_CAPTURE = 12000

# How long to ignore new voice commands after one fires, in seconds.
COMMAND_COOLDOWN_S = 2.0

# Max audio blocks to process per timer tick.
MAX_BLOCKS_PER_TICK = 5


class AudioTriggerNode(Node):

    def __init__(self):
        super().__init__('audio_trigger_node')

        self.wake_phrase = WAKE_PHRASE
        self.dismiss_phrase = DISMISS_PHRASE

        self.state_pub = self.create_publisher(String, '/smartbin/robot_state', 10)
        self.state_sub = self.create_subscription(
            String, '/smartbin/robot_state', self.external_state_callback, 10
        )

        self.audio_queue = queue.Queue()
        self.current_state = "IDLE"
        self._last_command_t = 0.0 # wall-clock time of the last fired command

        self.get_logger().info(f"Audio system initialized. State: {self.current_state}")

        model_path = os.path.expanduser(
            '~/smartbin_ws/src/smartbin_robot/models/vosk-model-small-en-us'
        )
        if not os.path.exists(model_path):
            self.get_logger().error(f"Vosk model not found at {model_path}.")
            sys.exit(1)

        self.model = vosk.Model(model_path)

        self.device_index = 0
        self.recognizer = vosk.KaldiRecognizer(self.model, VOSK_RATE)

        self.stream = sd.RawInputStream(
            samplerate=CAPTURE_RATE,
            blocksize=BLOCKSIZE_CAPTURE,
            device=self.device_index,
            dtype='int16',
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()
        self.get_logger().info(
            f"Listening for '{self.wake_phrase}' / '{self.dismiss_phrase}'..."
        )

        self.timer = self.create_timer(0.1, self.process_audio)

    def external_state_callback(self, msg):
        new_state = msg.data
        if new_state != self.current_state:
            self.get_logger().info(
                f"External state change: {self.current_state} -> {new_state}"
            )
            self.current_state = new_state

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            if 'overflow' in str(status).lower():
                self.get_logger().warn("Audio overflow — processing can't keep up.")
        # Downsample from 48kHz to 16kHz by taking every 3rd sample
        samples = np.frombuffer(indata, dtype=np.int16)
        downsampled = np.ascontiguousarray(samples[::DOWNSAMPLE])
        self.audio_queue.put(downsampled.tobytes())

    def process_audio(self):
        processed = 0
        while not self.audio_queue.empty() and processed < MAX_BLOCKS_PER_TICK:
            data = self.audio_queue.get()
            processed += 1
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    self.parse_command(text)
            else:
                partial = json.loads(self.recognizer.PartialResult()).get("partial", "").strip()
                if partial:
                    self.get_logger().debug(f"Partial: '{partial}'")

        # If the queue is still backed up after the cap, drain and reset the recognizer
        # so stale audio doesn't confuse future recognition.
        if self.audio_queue.qsize() > 10:
            self.get_logger().warn(
                f"Audio queue backed up ({self.audio_queue.qsize()} blocks) — flushing."
            )
            while not self.audio_queue.empty():
                self.audio_queue.get()
            self.recognizer = vosk.KaldiRecognizer(self.model, VOSK_RATE)

    def parse_command(self, text):
        # Ignore repeated firing within the cooldown window
        now = time.monotonic()
        if (now - self._last_command_t) < COMMAND_COOLDOWN_S:
            self.get_logger().debug(f"Cooldown active, ignoring: '{text}'")
            return

        self.get_logger().info(f"Heard: '{text}'")

        if self.wake_phrase in text:
            # Allow re-triggering from IDLE, or from a failed/stuck GOHOME/GOTARGET
            if self.current_state in ["IDLE", "GOHOME", "GOTARGET"]:
                self._last_command_t = now
                self._transition_to("SEARCH")
            elif self.current_state == "SEARCH":
                self.get_logger().info("Already searching.")
            else:
                self.get_logger().info(f"Ignored wake phrase — in '{self.current_state}' state.")

        elif self.dismiss_phrase in text:
            # Allow from any active state, always a valid escape
            if self.current_state != "IDLE":
                self._last_command_t = now
                self._transition_to("GOHOME")
            else:
                self.get_logger().info("Ignored dismiss phrase — already idle.")

    def _transition_to(self, new_state: str):
        self.get_logger().info(f"State transition: {self.current_state} -> {new_state}")
        self.current_state = new_state
        msg = String()
        msg.data = new_state
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AudioTriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stream.stop()
        node.stream.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()