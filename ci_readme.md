# Continuous Integration (CI) Pipelines for OpenVINO Build Deploy

## Table of Contents

- [Overview](#overview)
- [CI Pipelines](#ci-pipelines)
  - [Sanity Check (Notebooks)](#sanity-check-notebooks)
  - [Sanity Check (AI Ref Kits)](#sanity-check-ai-ref-kits)
  - [Sanity Check (Demos)](#sanity-check-demos)
  - [Stale Issues and Pull Requests](#stale-issues-and-pull-requests)
- [Reusable Steps](#reusable-steps)
  - [Find Updates](#find-updates)
  - [Categorize Projects](#categorize-projects)
  - [Setup OS](#setup-os)
  - [Setup Python](#setup-python)
  - [Gradio Action](#gradio-action)
  - [Timeouted Action](#timeouted-action)
- [Setup and Configuration](#setup-and-configuration)
  - [Prerequisites](#prerequisites)
  - [Running the Pipelines](#running-the-pipelines)
- [Troubleshooting](#troubleshooting)
- [Diagrams](#diagrams)
- [Collaboration and Feedback](#collaboration-and-feedback)

## Overview

This document provides an overview of the CI pipelines used in the OpenVINO Build Deploy repository. It includes setup and configuration details, step-by-step instructions for running the pipelines, troubleshooting common issues, and understanding the output.

## CI Pipelines

CI Pipelines are defined in the `.github/workflows` directory and the workflows performs sanity checks on demo projects and automatically manages and closes inactive issues.

### 1. Sanity Check (Notebooks)

- **File:** `.github/workflows/sanity-check-notebooks.yml`
- **Triggers:** 
  - Scheduled: Daily at 2 AM
  - Pull Requests to `master`
  - Pushes to `master`
  - Manual dispatch
- **Jobs:**
  - `find-subprojects`: Identifies subprojects to test.
  - `notebook`: Runs Jupyter notebooks on specified OS and Python versions.

#### Flow:
1. **Trigger:** The workflow is triggered by a schedule, pull request, push, or manual dispatch.
2. **find-subprojects Job:**
   - Checks out the code.
   - Determines which subprojects need to be tested based on changes.
   - Categorizes the subprojects.
3. **notebook Job:**
   - Runs only if there are notebooks to test.
   - Sets up the environment for each OS and Python version.
   - Executes the Jupyter notebooks.

### 2. Sanity Check (AI Ref Kits)

- **File:** `.github/workflows/sanity-check-kits.yml`
- **Triggers:** 
  - Scheduled: Daily at 2 AM
  - Pull Requests to `master`
  - Pushes to `master`
  - Manual dispatch
- **Jobs:**
  - `find-subprojects`: Identifies subprojects to test.
  - `qt`, `gradio`, `webcam`, `python`, `notebook`: Runs tests on specified OS and Python versions.

#### Flow:
1. **Trigger:** The workflow is triggered by a schedule, pull request, push, or manual dispatch.
2. **find-subprojects Job:**
   - Checks out the code.
   - Determines which subprojects need to be tested based on changes.
   - Categorizes the subprojects.
3. **qt, gradio, webcam, python, notebook Jobs:**
   - Each job runs only if there are corresponding subprojects to test.
   - Sets up the environment for each OS and Python version.
   - Executes the tests for each subproject type.

### 3. Sanity Check (Demos)

- **File:** `.github/workflows/sanity-check-demos.yml`
- **Triggers:** 
  - Scheduled: Daily at 2 AM
  - Pull Requests to `master`
  - Pushes to `master`
  - Manual dispatch
- **Jobs:**
  - `find-subprojects`: Identifies subprojects to test.
  - `gradio`, `webcam`, `js`: Runs tests on specified OS and Python versions.

#### Flow:
1. **Trigger:** The workflow is triggered by a schedule, pull request, push, or manual dispatch.
2. **find-subprojects Job:**
   - Checks out the code.
   - Determines which subprojects need to be tested based on changes.
   - Categorizes the subprojects.
3. **gradio, webcam, js Jobs:**
   - Each job runs only if there are corresponding subprojects to test.
   - Sets up the environment for each OS and Python version.
   - Executes the tests for each subproject type.

### 4. Stale Issues and Pull Requests

- **File:** `.github/workflows/stale.yml`
- **Triggers:** 
  - Scheduled: Daily at 2 AM
- **Jobs:**
  - `stale`: Marks issues and pull requests as stale after a period of inactivity and closes them if no further activity occurs.

#### Flow:
1. **Trigger:** The workflow is triggered by a schedule.
2. **stale Job:**
   - Identifies issues and pull requests that have been inactive.
   - Marks them as stale.
   - Closes them if no further activity occurs.

## Reusable Steps

Reusable steps are defined in the `.github/reusable-steps` directory and are used across multiple workflows to avoid duplication and ensure consistency.

### 1. Find Updates

- **File:** `.github/reusable-steps/find-updates/action.yml`
- **Description:** Identifies subprojects that have been updated based on changes in the repository.
- **Inputs:**
  - `dir`: Directory to check for updates.
  - `ci_config_file`: CI configuration file to consider for changes.
- **Outputs:**
  - `subproject_dirs`: List of updated subproject directories.

#### Flow:
1. **Trigger:** Called by workflows to identify updated subprojects.
2. **Steps:**
   - Checks for changes in the specified directory and CI configuration file.
   - Determines which subprojects have been updated.
   - Outputs the list of updated subproject directories.

### 2. Categorize Projects

- **File:** `.github/reusable-steps/categorize-projects/action.yml`
- **Description:** Categorizes subprojects based on their type.
- **Inputs:**
  - `subprojects`: List of subproject directories.
- **Outputs:**
  - `qt`, `gradio`, `webcam`, `python`, `notebook`: Categorized subproject directories.

#### Flow:
1. **Trigger:** Called by workflows to categorize subprojects.
2. **Steps:**
   - Categorizes the subprojects based on their type.
   - Outputs the categorized subproject directories.

### 3. Setup OS

- **File:** `.github/reusable-steps/setup-os/action.yml`
- **Description:** Sets up the operating system environment for the CI jobs.

#### Flow:
1. **Trigger:** Called by workflows to set up the OS environment.
2. **Steps:**
   - Sets up the necessary OS environment for the CI jobs.

### 4. Setup Python

- **File:** `.github/reusable-steps/setup-python/action.yml`
- **Description:** Sets up the Python environment for the CI jobs.
- **Inputs:**
  - `python`: Python version to set up.
  - `project`: Subproject directory.

#### Flow:
1. **Trigger:** Called by workflows to set up the Python environment.
2. **Steps:**
   - Sets up the specified Python version.
   - Configures the environment for the specified subproject.

### 5. Gradio Action

- **File:** `.github/reusable-steps/gradio-action/action.yml`
- **Description:** Runs Gradio demos.
- **Inputs:**
  - `script`: Script to execute.
  - `project`: Subproject directory.
  - `timeout`: Timeout duration.

#### Flow:
1. **Trigger:** Called by workflows to run Gradio demos.
2. **Steps:**
   - Executes the specified Gradio script.
   - Runs the demo within the specified timeout duration.

### 6. Timeouted Action

- **File:** `.github/reusable-steps/timeouted-action/action.yml`
- **Description:** Runs a command with a specified timeout.
- **Inputs:**
  - `command`: Command to execute.
  - `project`: Subproject directory.
  - `timeout`: Timeout duration.

#### Flow:
1. **Trigger:** Called by workflows to run a command with a timeout.
2. **Steps:**
   - Executes the specified command.
   - Ensures the command completes within the specified timeout duration.

## Setup and Configuration

### Prerequisites

- Basic knowledge of Git and GitHub Actions.
- Ensure you have the necessary permissions to trigger workflows and access secrets.
- Secrets such as `HF_TOKEN` and `GITHUB_TOKEN` should be configured in the repository settings.

### Running the Pipelines

1. **Automatic Triggers:**
   - Pipelines are triggered automatically based on the defined schedules, pull requests, and pushes to the `master` branch.

2. **Manual Triggers:**
   - Navigate to the Actions tab in the GitHub repository.
   - Select the desired workflow and click on "Run workflow".

3. **Pull Requests and Pushes:** 
   - Ensure the branch names match the triggers specified in the workflows.

## Troubleshooting

### Common Issues

- **Authentication Errors**: Ensure that all required secrets are correctly set in the repository settings.
- **Dependency Issues**: Verify that all dependencies are listed and compatible with the specified versions.
- **Timeouts**: Adjust the timeout settings in the workflows if certain steps consistently exceed the allotted time.

### Debugging Steps

1. **Check Logs**: Review the logs in the GitHub Actions tab for detailed error messages.
2. **Re-run Jobs**: Use the "Re-run jobs" feature in GitHub Actions to retry failed steps.
3. **Local Testing**: Test scripts and commands locally to ensure they work as expected before committing changes.

## Understanding the Output

- **Logs**: Each step in the workflow provides logs accessible in the GitHub Actions tab.
- **Status Badges**: Add status badges to your README to display the current status of your workflows.

## Diagrams

### CI Pipeline Flow
![CI-Flow](https://github.com/user-attachments/assets/e031fda7-a2e4-4a06-96f6-47b3f7289844)

For further assistance, please contact the project maintainers.
