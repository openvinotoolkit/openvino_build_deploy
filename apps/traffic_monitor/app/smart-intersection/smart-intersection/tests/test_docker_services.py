# SPDX-FileCopyrightText: (C) 2025 Intel Corporation
# SPDX-License-Identifier: LicenseRef-Intel-Edge-Software
# This file is licensed under the Limited Edge Software Distribution License Agreement.

import subprocess
import time
import pytest
import json

from .conftest import DOCKER_COMPOSE_FILE
from tests.utils.docker_utils import get_all_services, get_running_services

@pytest.mark.zephyr_id("NEX-T9367")
def test_docker_build_and_deployment():
  """Test that all docker-compose services are running after build and deploy."""
  running = get_running_services()
  expected = get_all_services()
  assert expected == running, f"Not all services are running. Expected: {expected}, Running: {running}"
