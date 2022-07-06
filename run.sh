pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
# uvicorn main:app --reload --port 8000 --host 0.0.0.0
uvicorn app:app --reload --port 8000 --host 0.0.0.0
