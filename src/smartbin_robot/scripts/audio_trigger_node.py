#!/usr/bin/env python3
import json
import queue
import sys
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd
import vosk

WAKE_PHRASE    = "robot come here"
DISMISS_PHRASE = "robot go home"

class AudioTriggerNode(Node):

    def __init__(self):
        super().__init__('audio_trigger_node')

        self.wake_phrase    = WAKE_PHRASE
        self.dismiss_phrase = DISMISS_PHRASE

        # Publisher: broadcasts state changes to the rest of the system
        self.state_pub = self.create_publisher(String, '/smartbin/robot_state', 10)

        # Subscriber: listens for state changes published by OTHER nodes
        # (e.g. human_detector_node publishing IDLE when the robot reaches home)
        # This keeps our local state in sync without needing a separate channel.
        self.state_sub = self.create_subscription(
            String, '/smartbin/robot_state', self.external_state_callback, 10
        )

        # Thread-safe queue to pass audio blocks from the stream callback to main loop
        self.audio_queue = queue.Queue()

        # Current state tracking
        self.current_state = "IDLE"
        self.get_logger().info(f"Audio system initialized. Current state: {self.current_state}")

        # Set up Vosk model path
        model_path = os.path.expanduser(
            '~/smartbin_ws/src/smartbin_robot/models/vosk-model-small-en-us'
        )
        if not os.path.exists(model_path):
            self.get_logger().error(f"Vosk model not found at {model_path}. Please check the path!")
            sys.exit(1)

        self.model = vosk.Model(model_path)

        # Audio stream parameters
        self.sample_rate  = 48000
        self.device_index = 0  # Change to your microphone index if needed

        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)

        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=24000,
            device=self.device_index,
            dtype='int16',
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()
        self.get_logger().info(
            f"Microphone stream started. "
            f"Listening for '{self.wake_phrase}' / '{self.dismiss_phrase}'..."
        )

        self.timer = self.create_timer(0.1, self.process_audio)

    # ------------------------------------------------------------------
    # External state sync
    # ------------------------------------------------------------------

    def external_state_callback(self, msg):
        """
        Receives state updates published by other nodes (e.g. human_detector_node
        publishing IDLE once the robot has reached home). Updates local state so
        we correctly gate future voice commands.
        """
        new_state = msg.data
        if new_state != self.current_state:
            self.get_logger().info(
                f"External state change received: {self.current_state} -> {new_state}"
            )
            self.current_state = new_state

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    def audio_callback(self, indata, frames, time, status):
        """Runs in a background thread; just enqueues raw audio bytes."""
        if status:
            self.get_logger().warn(str(status))
        self.audio_queue.put(bytes(indata))

    def process_audio(self):
        """Called by ROS 2 timer; drains the audio queue and feeds Vosk."""
        while not self.audio_queue.empty():
            data = self.audio_queue.get()
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                if text:
                    self.parse_command(text)
            else:
                partial = json.loads(self.recognizer.PartialResult()).get("partial", "")
                if partial:
                    self.get_logger().info(f"Thinking: '{partial}'...")

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def parse_command(self, text):
        self.get_logger().info(f"Heard: '{text}'")

        if self.wake_phrase in text:
            # "robot come here" — only valid from IDLE
            if self.current_state == "IDLE":
                self._transition_to("SEARCH")
            else:
                self.get_logger().info(
                    f"Ignored wake phrase — already in '{self.current_state}' state."
                )

        elif self.dismiss_phrase in text:
            # "robot go home" — valid from SEARCH or GOTARGET (not IDLE)
            if self.current_state in ["SEARCH", "GOTARGET"]:
                self._transition_to("GOHOME")
            else:
                self.get_logger().info("Ignored dismiss phrase — robot is already home/going home.")

    def _transition_to(self, new_state: str):
        self.get_logger().info(
            f"State transition: {self.current_state} -> {new_state}"
        )
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