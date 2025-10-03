# SPDX-FileCopyrightText: (C) 2025 Intel Corporation
# SPDX-License-Identifier: LicenseRef-Intel-Edge-Software
# This file is licensed under the Limited Edge Software Distribution License Agreement.

import pytest
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from tests.utils.ui_utils import waiter, driver
from tests.utils.utils import check_components_access
from .conftest import (
  SCENESCAPE_URL,
  GRAFANA_URL,
  INFLUX_DB_URL,
  NODE_RED_URL,
  INFLUX_DB_ADMIN_USERNAME,
  INFLUX_DB_ADMIN_PASSWORD
)

@pytest.mark.zephyr_id("NEX-T9368")
def test_components_access():
  """Test that all application components are accessible."""
  urls_to_check = [
    SCENESCAPE_URL,
    GRAFANA_URL,
    INFLUX_DB_URL,
    NODE_RED_URL
  ]

  for url in urls_to_check:
    check_components_access(url)

@pytest.mark.zephyr_id("NEX-T9623")
def test_grafana_failed_login(waiter):
  waiter.perform_login(
    GRAFANA_URL,
    By.CSS_SELECTOR, "[data-testid='data-testid Username input field']",
    By.CSS_SELECTOR, "[data-testid='data-testid Password input field']",
    By.CSS_SELECTOR, "[data-testid='data-testid Login button']",
    "wrong_username", "wrong_password"
  )

  # Wait for the error message element to appear
  waiter.wait_and_assert(
    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='data-testid Alert error']")),
    error_message="Login error message not found within 10 seconds"
  )

@pytest.mark.zephyr_id("NEX-T9617")
def test_influx_db_login(waiter):
  waiter.perform_login(
    INFLUX_DB_URL,
    By.CSS_SELECTOR, "[data-testid='username']",
    By.CSS_SELECTOR, "[data-testid='password']",
    By.CSS_SELECTOR, "[data-testid='button']",
    INFLUX_DB_ADMIN_USERNAME, INFLUX_DB_ADMIN_PASSWORD
  )

    # Wait for the header element to be visible after login
  waiter.wait_and_assert(
    EC.visibility_of_element_located((By.CSS_SELECTOR, "[data-testid='home-page--header']")),
    error_message='Welcome message not visible within 10 seconds after login'
  )

@pytest.mark.zephyr_id("NEX-T9621")
def test_influx_db_failed_login(waiter):
  waiter.perform_login(
    INFLUX_DB_URL,
    By.CSS_SELECTOR, "[data-testid='username']",
    By.CSS_SELECTOR, "[data-testid='password']",
    By.CSS_SELECTOR, "[data-testid='button']",
    "wrong_username", "wrong_password"
  )

  # Wait for the error notification to be visible after failed login
  waiter.wait_and_assert(
    EC.visibility_of_element_located((By.CSS_SELECTOR, "[data-testid='notification-error--children']")),
    error_message='Error notification not visible within 10 seconds after failed login'
  )
