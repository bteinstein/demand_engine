import subprocess
import time
import requests
from pathlib import Path
import logging

class ValhallaManager:
    """Manages the Valhalla Docker container lifecycle."""

    def __init__(self, compose_dir=None, compose_file=None, valhalla_url="http://localhost:8002", initial_delay=10, logger=logging.getLogger(__name__)):
        """Initialize the ValhallaManager with paths and configuration."""
        self.compose_dir = Path(compose_dir or Path.home() / "BT" / "docker_home" / "valhalla_nigeria_project")
        self.compose_file = Path(compose_file or self.compose_dir / "docker-compose.yml")
        self.valhalla_url = valhalla_url
        self.valhalla_status_url = valhalla_url+ "/status"
        self.initial_delay = initial_delay  # Seconds to wait after starting the container
        self.logger = logger   

    def run_command(self, command, cwd=None):
        """Run a shell command with sudo and return the output, suppressing warnings."""
        try:
            result = subprocess.run(
                f"sudo {command}",
                shell=True,
                check=True,
                text=True,
                capture_output=True,
                cwd=cwd or self.compose_dir
            )
            # Filter out warnings from stderr
            stderr = result.stderr
            if "the attribute `version` is obsolete" not in stderr:
                print(stderr)  # Only print stderr if it's not the version warning
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running command '{command}': {e.stderr}")
            raise

    def start_valhalla(self):
        """Start the Valhalla Docker container and wait for initialization."""
        print("Starting Valhalla container...")
        command = f"docker-compose -f {self.compose_file} up -d --build"
        self.run_command(command)
        print(f"Waiting {self.initial_delay} seconds for Valhalla to initialize...")
        time.sleep(self.initial_delay)
        print("Valhalla container started.")

    def check_valhalla_status(self, max_attempts=5, delay=5):
        """Check if the Valhalla server is running by querying its status endpoint."""
        self.logger.debug("Checking Valhalla server status...")
        for attempt in range(max_attempts):
            try:
                response = requests.get(self.valhalla_status_url, timeout=5)
                if response.status_code == 200:
                    self.logger.debug("Valhalla server is running.")
                    return True
                else:
                    self.logger.debug(f"Attempt {attempt + 1}/{max_attempts}: Server returned {response.status_code}")
            except requests.RequestException as e:
                self.logger.debug(f"Attempt {attempt + 1}/{max_attempts}: Failed to connect - {e}")
            
            if attempt < max_attempts - 1:
                self.logger.debug(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
        
        self.logger.info("Valhalla server failed to start.")
        return False

    def stop_valhalla(self):
        """Stop the Valhalla Docker container."""
        print("Stopping Valhalla container...")
        command = f"docker-compose -f {self.compose_file} down"
        self.run_command(command)
        print("Valhalla container stopped.")

def main():
    # Instantiate the ValhallaManager
    manager = ValhallaManager()

    # Example usage of independent methods
    try:
        # Start the server
        manager.start_valhalla()

        # Verify the server is running
        if manager.check_valhalla_status():
            print("Server is up, performing custom task...")
            # Example task: Simulate some work
            time.sleep(10)  # Replace with your actual task
            print("Custom task completed.")
        else:
            print("Failed to verify Valhalla server. Exiting...")
            return

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Stop the server
        manager.stop_valhalla()

if __name__ == "__main__":
    main()