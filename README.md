

<a href="https://doi.org/10.1371/journal.pone.0296545">
<img src="https://journals.plos.org/resource/img/one/logo.png" style="background-color:white;height:45px;">

[![](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/soilspectroscopy)

[![](https://zenodo.org/badge/doi/10.5281/zenodo.5759693.svg)](https://doi.org/10.5281/zenodo.5759693)

Welcome to the OSSL chemometrics manual!

This is a code repository, please visit the
[website](https://soilspectroscopy.github.io/ossl-chemometrics/) to
navigate through the different content sections.

A [peer-reviewed and open-access
publication](https://doi.org/10.1371/journal.pone.0296545) on the OSSL
project is available for additional reference.

You can also visit the [OSSL manual](https://docs.soilspectroscopy.org/)
and other additional resources in our [GitHub
organization](https://github.com/soilspectroscopy).

## Local Setup

To run this application locally, you will need **Python 3.12**.

### 1. Clone the repository

``` bash
git clone [https://github.com/soilspectroscopy/ossl-chemometrics.git](https://github.com/soilspectroscopy/ossl-chemometrics.git)
cd ossl-chemometrics
```

### 2. Create and Activate the Environment

``` bash
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the App

As a regular Shiny app:

``` bash
shiny run app.py --port 8080 --reload
kill -9 $(lsof -t -i:8080)  # To close the server
```

As a Shinylive app:

``` bash
shinylive export . _site
python3 -m http.server --directory _site --bind localhost 8008
pkill -f "http.server"  # To close the server
```
