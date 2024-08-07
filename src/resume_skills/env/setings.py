from dotenv import load_dotenv
import logging
from pathlib import Path
from os import getenv

# Path
REPOSITORY_PATH = Path(__file__).parents[3]


# --> Env variables
load_dotenv(Path(REPOSITORY_PATH, ".env"))

# --> Logging settings
log_format = "%(asctime)s : %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")

OPEN_API_KEY = getenv("OPENAI_API_KEY")
