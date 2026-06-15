import concurrent.futures.process
import sentence_transformers
import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())
