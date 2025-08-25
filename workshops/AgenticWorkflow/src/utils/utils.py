from dotenv import load_dotenv

def load_env():
    """
    Load environment variables from a .env file.
    This function is used to load the environment variables required for the application.
    """
    load_dotenv()
    return