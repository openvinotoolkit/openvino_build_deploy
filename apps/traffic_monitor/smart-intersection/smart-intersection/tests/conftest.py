# SPDX-FileCopyrightText: (C) 2025 Intel Corporation
# SPDX-License-Identifier: LicenseRef-Intel-Edge-Software
# This file is licensed under the Limited Edge Software Distribution License Agreement.

import pytest
import logging
import time
import os
import subprocess
import requests
from dotenv import load_dotenv
from tests.utils.utils import (
  run_command,
  check_components_access,
  get_password_from_supass_file,
  get_username_from_influxdb2_admin_username_file,
  get_password_from_influxdb2_admin_password_file
)

load_dotenv()
logger = logging.getLogger(__name__)

DOCKER_COMPOSE_FILE = os.getenv("DOCKER_COMPOSE_FILE", "compose.yml")

SCENESCAPE_URL = os.getenv("SCENESCAPE_URL", "https://localhost")
SCENESCAPE_USERNAME = os.getenv("SCENESCAPE_USERNAME", "admin")
SCENESCAPE_PASSWORD = os.getenv("SCENESCAPE_PASSWORD", get_password_from_supass_file())

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USERNAME = os.getenv("GRAFANA_USERNAME", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "admin")

INFLUX_DB_URL = os.getenv("INFLUX_DB_URL", "http://localhost:8086")
INFLUX_DB_ADMIN_USERNAME = os.getenv("INFLUX_DB_ADMIN_USERNAME", get_username_from_influxdb2_admin_username_file())
INFLUX_DB_ADMIN_PASSWORD = os.getenv("INFLUX_DB_ADMIN_PASSWORD", get_password_from_influxdb2_admin_password_file())

NODE_RED_URL = os.getenv("NODE_RED_URL", "http://localhost:1880")

def wait_for_services_readiness(services_urls, timeout=120, interval=2):
  """
  Waits for services to become available.

  :param services_urls: List of URLs to check.
  :param timeout: Maximum time to wait for services to be available (in seconds).
  :param interval: Time between checks (in seconds).
  :raises TimeoutError: If services are not available within the timeout period.
  """
  start_time = time.time()
  while time.time() - start_time < timeout:
    all_services_ready = True
    for url in services_urls:
      try:
        check_components_access(url)
      except AssertionError as e:
        all_services_ready = False
        break
    if all_services_ready:
      logger.info("All services are ready.")
      return True
    logger.info("Waiting for services to be ready...")
    time.sleep(interval)
  raise TimeoutError("Services did not become ready in time.")

@pytest.fixture(scope="session", autouse=True)
def build_and_deploy():
  """
  Fixture to build and deploy Docker containers for testing.

  This fixture is automatically used for the entire test session.
  """
  # Build Docker images
  out, err, code = run_command(f"docker compose -f {DOCKER_COMPOSE_FILE} build")
  assert code == 0, f"Build failed: {err}"

  # Deploy (up) Docker containers
  out, err, code = run_command(f"docker compose -f {DOCKER_COMPOSE_FILE} up -d")
  assert code == 0, f"Deploy failed: {err}"

  # Wait for services to be ready
  services_urls = [SCENESCAPE_URL, GRAFANA_URL, INFLUX_DB_URL, NODE_RED_URL]
  wait_for_services_readiness(services_urls)

  yield

  # Teardown: stop and remove containers
  run_command(f"docker compose -f {DOCKER_COMPOSE_FILE} down")
  # Remove Docker volumes
  run_command("docker volume ls | grep smart-intersection | awk '{ print $2 }' | xargs docker volume rm")
