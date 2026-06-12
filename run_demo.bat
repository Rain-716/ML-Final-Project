@echo off
pip install -r requirements.txt
python src\train.py --data_source sample
python src\app.py
