# SPDX-FileCopyrightText: (C) 2025 Intel Corporation
# SPDX-License-Identifier: LicenseRef-Intel-Edge-Software
# This file is licensed under the Limited Edge Software Distribution License Agreement.

import pytest
import logging
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from tests.utils.ui_utils import waiter, driver
from tests.utils.utils import check_components_access
from .conftest import SCENESCAPE_URL, SCENESCAPE_PASSWORD, SCENESCAPE_USERNAME

logger = logging.getLogger(__name__)

@pytest.mark.zephyr_id("NEX-T9389")
def test_login(waiter):
  """Test that the admin login functionality works correctly."""
  waiter.perform_login(
    SCENESCAPE_URL,
    By.ID, "username",
    By.ID, "password",
    By.ID, "login-submit",
    SCENESCAPE_USERNAME, SCENESCAPE_PASSWORD
  )

  # Verify that the expected elements are present on the page
  nav_scenes = waiter.wait_and_assert(
    EC.presence_of_element_located((By.ID, "nav-scenes")),
    error_message='"nav-scenes" element not found on the page'
  )

  assert nav_scenes

@pytest.mark.zephyr_id("NEX-T9390")
def test_logout(waiter):
  """Test that the admin logout functionality works correctly."""
  waiter.perform_login(
    SCENESCAPE_URL,
    By.ID, "username",
    By.ID, "password",
    By.ID, "login-submit",
    SCENESCAPE_USERNAME, SCENESCAPE_PASSWORD
  )

  # Perform logout action
  logout_link = waiter.wait_and_assert(
    EC.presence_of_element_located((By.ID, "nav-sign-out")),
    error_message='"nav-sign-out" element not found on the page'
  )
  logout_link.click()

  # Wait for the 'username' input to be present
  waiter.wait_and_assert(
    EC.presence_of_element_located((By.ID, "username")),
    error_message='"username" input field not found within 10 seconds'
  )

@pytest.mark.zephyr_id("NEX-T9388")
def test_change_password(waiter):
  """Test that the admin can change the password successfully."""
  waiter.perform_login(
    SCENESCAPE_URL,
    By.ID, "username",
    By.ID, "password",
    By.ID, "login-submit",
    SCENESCAPE_USERNAME, SCENESCAPE_PASSWORD
  )

  # Navigate to Password change page
  waiter.driver.get(SCENESCAPE_URL + "/admin/password_change")

  # Wait for the 'Change my password' button to be present
  change_password_button = waiter.wait_and_assert(
    EC.presence_of_element_located((By.XPATH, "//input[@type='submit' and @value='Change my password']")),
    error_message='"Change my password" button not found within 10 seconds'
  )

  old_password_input = waiter.driver.find_element(By.ID, "id_old_password")
  new_password1_input = waiter.driver.find_element(By.ID, "id_new_password1")
  new_password2_input = waiter.driver.find_element(By.ID, "id_new_password2")

  old_password_input.send_keys(SCENESCAPE_PASSWORD)
  new_password1_input.send_keys(SCENESCAPE_PASSWORD)
  new_password2_input.send_keys(SCENESCAPE_PASSWORD)

  # Submit the password change
  change_password_button.click()

  # Wait for the success message to be present
  waiter.wait_and_assert(
    EC.text_to_be_present_in_element((By.TAG_NAME, "body"), "Password change successful"),
    error_message='"Password change successful" message not found within 10 seconds'
  )

@pytest.mark.zephyr_id("NEX-T9374")
def test_web_options_availability(waiter):
  """Test that the web option is available in the admin interface."""
  waiter.perform_login(
    SCENESCAPE_URL,
    By.ID, "username",
    By.ID, "password",
    By.ID, "login-submit",
    SCENESCAPE_USERNAME, SCENESCAPE_PASSWORD
  )

  # Define static list of navbar links
  navbar_links = [
    SCENESCAPE_URL + "/",  # Scenes
    SCENESCAPE_URL + "/cam/list/",  # Cameras
    SCENESCAPE_URL + "/singleton_sensor/list/",  # Sensors
    SCENESCAPE_URL + "/asset/list/",  # Object Library
    "https://docs.openedgeplatform.intel.com/scenescape/main/toc.html",  # Documentation
    SCENESCAPE_URL + "/admin"  # Admin
  ]

  # Check each link for a 200 status code
  for url in navbar_links:
    logger.info("Checking URL: %s", url)
    check_components_access(url)
