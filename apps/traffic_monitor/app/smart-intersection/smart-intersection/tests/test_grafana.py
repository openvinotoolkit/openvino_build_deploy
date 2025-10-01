# SPDX-FileCopyrightText: (C) 2025 Intel Corporation
# SPDX-License-Identifier: LicenseRef-Intel-Edge-Software
# This file is licensed under the Limited Edge Software Distribution License Agreement.

import pytest
import logging
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from tests.utils.ui_utils import waiter, driver
from .conftest import GRAFANA_URL, GRAFANA_USERNAME, GRAFANA_PASSWORD

logger = logging.getLogger(__name__)

def check_grafana_panel_value(waiter):
  """Check the value of one of the Grafana panels."""

  # Wait for the section to be present
  section = waiter.wait_and_assert(
    EC.presence_of_element_located((By.CSS_SELECTOR, "section[data-testid='data-testid Panel header South Bound lane - dwell time']")),
    error_message="Panel section is not present on the page"
    )

  # Find the specific span within the section
  span_element = section.find_element(By.CSS_SELECTOR, "span.flot-temp-elem")

  # Assert that the value is not "No data"
  assert span_element.text != "No data", "No data is displayed in the span element"


@pytest.mark.zephyr_id("NEX-T9371")
def test_grafana_anthem_dashboard_availability(waiter):
  """Test the availability of the Anthem dashboard in Grafana."""  
  waiter.perform_login(
    GRAFANA_URL,
    By.CSS_SELECTOR, "[data-testid='data-testid Username input field']",
    By.CSS_SELECTOR, "[data-testid='data-testid Password input field']",
    By.CSS_SELECTOR, "[data-testid='data-testid Login button']",
    GRAFANA_USERNAME, GRAFANA_PASSWORD
  )

  skip_button = waiter.wait_and_assert(
    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='data-testid Skip change password button']")),
    error_message="Skip button is not present on the page"
  )
  skip_button.click()
      
  # Find and click the "Dashboards" link
  dashboards_link = waiter.wait_and_assert(
    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='data-testid Nav menu item'][href='/dashboards']")),
    error_message="Dashboards link is not present on the page"
  )
  dashboards_link.click()

  # Find and click the link to the "Anthem-ITS-Data" dashboard
  anthem_dashboard_link = waiter.wait_and_assert(
    EC.presence_of_element_located((By.LINK_TEXT, "Anthem-ITS-Data")),
    error_message="Anthem-ITS-Data dashboard link is not present on the page"
  )
  anthem_dashboard_link.click()

  check_grafana_panel_value(waiter)
