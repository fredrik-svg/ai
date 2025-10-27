"""
MQTT Handler Module for communication with n8n workflows.
Handles sending user input and receiving responses via HiveMQ Cloud.
"""

import logging
import ssl
import json
import time
from typing import Optional, Callable
import paho.mqtt.client as mqtt


class MQTTHandler:
    """
    MQTT client for communicating with n8n workflows via HiveMQ Cloud.
    """

    def __init__(
        self,
        broker: str,
        port: int = 8883,
        username: str = "",
        password: str = "",
        topic_send: str = "assistant/input",
        topic_receive: str = "assistant/output",
        use_tls: bool = True,
        keepalive: int = 60,
        qos: int = 1
    ):
        """
        Initialize the MQTT handler.

        Args:
            broker: MQTT broker address (e.g., 'cluster.hivemq.cloud')
            port: MQTT port (8883 for TLS, 1883 for non-TLS)
            username: MQTT username
            password: MQTT password
            topic_send: Topic to send messages to n8n
            topic_receive: Topic to receive responses from n8n
            use_tls: Whether to use TLS/SSL
            keepalive: Keep-alive interval in seconds
            qos: Quality of Service level (0, 1, or 2)
        """
        self.logger = logging.getLogger(__name__)
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.topic_send = topic_send
        self.topic_receive = topic_receive
        self.use_tls = use_tls
        self.keepalive = keepalive
        self.qos = qos

        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.response_callback: Optional[Callable[[str], None]] = None
        self.last_response: Optional[str] = None

        self.logger.info(f"MQTT Handler initialized for broker: {broker}:{port}")

    def connect(self) -> bool:
        """
        Connect to the MQTT broker.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create MQTT client
            self.client = mqtt.Client(client_id=f"assistant_{int(time.time())}")

            # Set username and password
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)

            # Configure TLS if needed
            if self.use_tls:
                self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)

            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message

            # Connect to broker
            self.logger.info(f"Connecting to MQTT broker {self.broker}:{self.port}...")
            self.client.connect(self.broker, self.port, self.keepalive)

            # Start the network loop in a background thread
            self.client.loop_start()

            # Wait for connection (with timeout)
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if self.connected:
                self.logger.info("Successfully connected to MQTT broker")
                return True
            else:
                self.logger.error("Failed to connect to MQTT broker (timeout)")
                return False

        except Exception as e:
            self.logger.error(f"Error connecting to MQTT broker: {e}")
            return False

    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when the client connects to the broker."""
        if rc == 0:
            self.connected = True
            self.logger.info("Connected to MQTT broker")
            
            # Subscribe to the response topic
            self.client.subscribe(self.topic_receive, qos=self.qos)
            self.logger.info(f"Subscribed to topic: {self.topic_receive}")
        else:
            self.connected = False
            self.logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects from the broker."""
        self.connected = False
        if rc != 0:
            self.logger.warning(f"Unexpected disconnection from MQTT broker. Return code: {rc}")
        else:
            self.logger.info("Disconnected from MQTT broker")

    def _on_message(self, client, userdata, msg):
        """Callback for when a message is received."""
        try:
            payload = msg.payload.decode('utf-8')
            self.logger.info(f"Received message on topic '{msg.topic}': {payload}")

            # Try to parse as JSON
            try:
                data = json.loads(payload)
                response_text = data.get('response', data.get('text', payload))
            except json.JSONDecodeError:
                # If not JSON, use the raw payload
                response_text = payload

            self.last_response = response_text

            # Call the response callback if set
            if self.response_callback:
                self.response_callback(response_text)

        except Exception as e:
            self.logger.error(f"Error processing received message: {e}")

    def send_message(self, text: str) -> bool:
        """
        Send a text message to n8n via MQTT.

        Args:
            text: Text message to send

        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.connected or self.client is None:
            self.logger.error("Not connected to MQTT broker")
            return False

        try:
            # Create message payload
            payload = json.dumps({
                "text": text,
                "timestamp": time.time()
            })

            # Publish message
            result = self.client.publish(self.topic_send, payload, qos=self.qos)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.info(f"Sent message to topic '{self.topic_send}': {text}")
                return True
            else:
                self.logger.error(f"Failed to send message. Return code: {result.rc}")
                return False

        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False

    def set_response_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set a callback function to be called when a response is received.

        Args:
            callback: Function to call with the response text
        """
        self.response_callback = callback
        self.logger.debug("Response callback set")

    def get_last_response(self) -> Optional[str]:
        """
        Get the last received response.

        Returns:
            Last response text, or None if no response received
        """
        return self.last_response

    def clear_last_response(self) -> None:
        """Clear the last response."""
        self.last_response = None

    def wait_for_response(self, timeout: float = 10.0) -> Optional[str]:
        """
        Wait for a response from n8n.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Response text, or None if timeout
        """
        self.clear_last_response()
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            if self.last_response is not None:
                return self.last_response
            time.sleep(0.1)

        self.logger.warning(f"Timeout waiting for response after {timeout} seconds")
        return None

    def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            self.logger.info("Disconnected from MQTT broker")

    def is_connected(self) -> bool:
        """
        Check if connected to the MQTT broker.

        Returns:
            True if connected, False otherwise
        """
        return self.connected

    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()
