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

WAKE_PHRASE = "robot come here"

class AudioTriggerNode(Node):

    def __init__(self):
        super().__init__('audio_trigger_node')

        self.wake_phrase = WAKE_PHRASE
        
        # ROS 2 Publisher for robot state
        self.state_pub = self.create_publisher(String, '/smartbin/robot_state', 10)
        
        # Thread-safe queue to pass audio blocks from the stream callback to main loop
        self.audio_queue = queue.Queue()
        
        # Current State tracking
        self.current_state = "IDLE"
        self.get_logger().info(f"Audio system initialized. Current state: {self.current_state}")

        # Set up Vosk Model Path
        model_path = os.path.expanduser('~/smartbin_ws/src/smartbin_robot/models/vosk-model-small-en-us')
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"Vosk model not found at {model_path}. Please check the path!")
            sys.exit(1)
            
        # Initialize Vosk Model
        self.model = vosk.Model(model_path)
        
        # Configure the audio stream parameters
        self.sample_rate = 48000 # Vosk expects 16kHz
        self.device_index = 0  # Uses audeze maxwell in my case, change to your microphone index if needed
        
        # Initialize the speech recognizer
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
        
        # Start the microphone audio stream
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate, 
            blocksize=24000, 
            device=self.device_index, 
            dtype='int16',
            channels=1, 
            callback=self.audio_callback
        )
        self.stream.start()
        self.get_logger().info(f"Microphone stream started. Listening for '{self.wake_phrase}'...")

        # Create a timer to process the audio queue at regular intervals
        self.timer = self.create_timer(0.1, self.process_audio)

    def audio_callback(self, indata, frames, time, status):
        """This callback runs in a separate background thread for every audio block captured."""
        if status:
            self.get_logger().warn(str(status))
        self.audio_queue.put(bytes(indata))

    def process_audio(self):
        """Main processing loop triggered by the ROS 2 timer."""
        while not self.audio_queue.empty():
            data = self.audio_queue.get()
            
            # Feed the raw PCM data into the Vosk recognizer
            if self.recognizer.AcceptWaveform(data):
                # AcceptWaveform returns True when a phrase/silence boundary is reached
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                if text:
                    self.parse_command(text)
            else:
                # Partial results can be read here if you want real-time tracking,
                partial_result = json.loads(self.recognizer.PartialResult())
                partial_text = partial_result.get("partial", "")
                if partial_text:
                    self.get_logger().info(f"Thinking: '{partial_text}'...")

    def parse_command(self, text):
        self.get_logger().info(f"Heard phrase: '{text}'")
        
        # Normalize text and check for the wake phrase
        # Vosk strips punctuation completely, so look for the wake phrase
        if self.wake_phrase in text:
            if self.current_state == "IDLE":
                self.current_state = "SEARCH"
                self.get_logger().info("Wake word matched! Transitioning state: IDLE -> SEARCH")
                
                # Publish the new state to the rest of the ROS 2 system
                msg = String()
                msg.data = self.current_state
                self.state_pub.publish(msg)
            else:
                self.get_logger().info(f"Ignored wake phrase because robot is already in {self.current_state} state.")

def main(args=None):
    rclpy.init(args=args)
    node = AudioTriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop audio hardware streams first
        node.stream.stop()
        node.stream.close()
        # Destroy the node clean and safe
        node.destroy_node()
        
        # Only call shutdown if the context is still active
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()