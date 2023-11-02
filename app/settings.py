import os
from dotenv import load_dotenv

dotenv_path = '.env'
load_dotenv(dotenv_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")