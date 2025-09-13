# SPDX-FileCopyrightText: (C) 2025 Intel Corporation
# SPDX-License-Identifier: LicenseRef-Intel-Edge-Software
# This file is licensed under the Limited Edge Software Distribution License Agreement.

import pytest
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from tests.utils.ui_utils import waiter, driver
from .conftest import SCENESCAPE_URL, SCENESCAPE_USERNAME, SCENESCAPE_PASSWORD


def add_camera(waiter, camera_name, camera_id):
  """Helper function to log in and add a new camera."""
  waiter.perform_login(
    SCENESCAPE_URL,
    By.ID, "username",
    By.ID, "password",
    By.ID, "login-submit",
    SCENESCAPE_USERNAME, SCENESCAPE_PASSWORD
  )

  # Find the 'Cameras' navigation link and click it
  cameras_nav_link = waiter.wait_and_assert(
    EC.presence_of_element_located((By.ID, "nav-cameras")),
    error_message="Cameras navigation link is not present on the page"
  )
  cameras_nav_link.click()

  # Find the 'New Camera' button and click it
  new_camera_button = waiter.wait_and_assert(
    EC.presence_of_element_located((By.XPATH, "//a[contains(@class, 'btn') and @href='/cam/create/']")),
    error_message="New Camera button is not present on the page"
  )
  new_camera_button.click()

  # Verify that the 'Add New Camera' button is present
  add_new_camera_button = waiter.wait_and_assert(
    EC.presence_of_element_located((By.XPATH, "//input[@class='btn btn-primary' and @value='Add New Camera']")),
    error_message="Add New Camera button is not present on the page"
  )

  # Fill new camera form fields
  sensor_id_input = waiter.driver.find_element(By.ID, "id_sensor_id")
  id_name_input = waiter.driver.find_element(By.ID, "id_name")
  id_scene_select = waiter.driver.find_element(By.ID, "id_scene")

  sensor_id_input.send_keys(camera_id)
  id_name_input.send_keys(camera_name)
  select = Select(id_scene_select)
  select.select_by_visible_text("Intersection-Demo")

  # Click the 'Add New Camera' button
  add_new_camera_button.click()

  # Verify that the new camera card is present
  camera_card = waiter.wait_and_assert(
    EC.visibility_of_element_located((By.ID, f"rate-{camera_id}")),
    error_message=f"Camera card with ID 'rate-{camera_id}' is not visible on the page"
  )
  return camera_card

@pytest.mark.zephyr_id("NEX-T9382")
def test_add_camera(waiter):
  """Test that the admin can add a new camera."""
  name_of_new_camera = "cam_NEX-T9382"
  id_of_new_camera = "cam_id_NEX-T9382"

  add_camera(waiter, name_of_new_camera, id_of_new_camera)

@pytest.mark.zephyr_id("NEX-T9383")
def test_delete_camera(waiter):
  """Test that the admin can delete a new camera."""
  name_of_new_camera = "cam_NEX-T9383"
  id_of_new_camera = "cam_id_NEX-T9383"

  camera_card = add_camera(waiter, name_of_new_camera, id_of_new_camera)

  # Find the 'Delete' button for the specific camera and click it
  delete_button = waiter.wait_and_assert(
    EC.presence_of_element_located((By.XPATH, f"//a[@title='Delete {name_of_new_camera}']")),
    error_message=f"Delete button for camera '{name_of_new_camera}' is not present on the page"
  )
  delete_button.click()

  # Verify that the confirmation button 'Yes, Delete the Camera!' is present and click it
  confirm_delete_button = waiter.wait_and_assert(
    EC.presence_of_element_located((By.XPATH, "//input[@class='btn btn-primary' and @value='Yes, Delete the Camera!']")),
    error_message="Confirmation button 'Yes, Delete the Camera!' is not present on the page"
  )
  confirm_delete_button.click()

  # Verify that the camera card is no longer present
  waiter.wait_and_assert(
    EC.invisibility_of_element_located((By.ID, f"rate-{id_of_new_camera}")),
    error_message=f"Camera card with ID 'rate-{id_of_new_camera}' is still visible on the page after deletion"
  )
