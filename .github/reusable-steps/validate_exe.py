import os
import subprocess
import argparse


def validate_exe(file_path):
    """
    This function checks if the .exe file exists and executes correctly.

    Args:
        file_path (str): The path to the .exe file to be validated.

    Returns:
        bool: True if the .exe file exists and executes correctly, False otherwise.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Execute the .exe file and capture the output
    result = subprocess.run([file_path], capture_output=True, text=True)

    if result.returncode == 0:
        print("Execution successful!")
        print("Output:")
        print(result.stdout)
        return True
    else:
        print("Execution failed!")
        print("Error Output:")
        print(result.stderr)
        raise RuntimeError("Executable failed to run")


parser = argparse.ArgumentParser(description='Path to executable file')
parser.add_argument('-p', '--path_exe', type=str, help="Path to executable file to evaluate")
args = parser.parse_args()


if __name__ == "__main__":
    if validate_exe(args.path_exe):
        print("Validation successful: The .exe file was created and executed correctly.")
    else:
        print("Validation failed: There was an issue with the .exe file.")
