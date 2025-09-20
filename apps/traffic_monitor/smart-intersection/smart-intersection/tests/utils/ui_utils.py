# SPDX-FileCopyrightText: (C) 2025 Intel Corporation
# SPDX-License-Identifier: LicenseRef-Intel-Edge-Software
# This file is licensed under the Limited Edge Software Distribution License Agreement.


import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from tests.utils.element_waiter import ElementWaiter

@pytest.fixture
def driver():
  chrome_options = Options()
  chrome_options.add_argument("--headless")  # Enables running Chrome without a graphical user interface
  chrome_options.add_argument("--no-sandbox")  # Disables process isolation, allowing broader system access
  chrome_options.add_argument("--disable-dev-shm-usage")  # Uses disk space for temporary files instead of shared memory
  chrome_options.add_argument("--ignore-certificate-errors") # Ignore SSL certificate errors
  chrome_options.add_argument("--allow-insecure-localhost") # Allow insecure connections to localhost
  chrome_options.add_argument("--window-size=1920,1080") # Set window size to ensure elements are visible
  service = Service(ChromeDriverManager().install())  # Automatically installs the appropriate ChromeDriver
  driver = webdriver.Chrome(service=service, options=chrome_options)
  yield driver
  driver.quit()

@pytest.fixture
def waiter(driver):
    return ElementWaiter(driver)
