### Technical assessment
This trial of the assessment is based on PCA dimensionality reduction, and DBSCAN clustering algorithm. For details of the methodology and results, please refer to the document.

### Initialization
Open a virtual environment and install the dependecies packages there.
```
python3 -m venv venv
. venv/bin/activuate
pip install -r requirements.txt
```

### Run the Code
Run the code in the foowing order
```
python3 preprocess.py # generate preprocessed csv
python3 cluster.py # output in /output
```