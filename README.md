# SVM practical assignment 

*(French version below)*

This short Git repository has a single objective: to learn how Support Vector Machines (SVM) work, using various related Python libraries.

This assignment will be supervised and graded by:

- BENSAID Bilel (bilel.bensaid@umontpellier.fr)

This assignment will be carried out exclusively by:

- CLETZ Laura (laura.cletz@etu.umontpellier.fr)

## Installation

### Prerequisites
- Python 3.8+
- Conda or pip package manager

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or with conda
conda install --file requirements.txt
```

## Usage

### Run Python scripts
```bash
cd python_script/
python svm_script.py
python svm_gui.py
```

### Generate Quarto report

We recommend producing an HTML file rather than a PDF.

```bash
cd quarto_report/
quarto render svm_report.qmd --to html
# quarto render svm_report.qmd --to pdf
```

---

# TP de mise en pratique des SVM

Ce court dépôt Git a un seul objectif : apprendre comment fonctionnent les Machines à Vecteurs Supports (SVM), à l'aide de diverses bibliothèques Python associées.

Cet exercice sera supervisé et noté par :

- BENSAID Bilel (bilel.bensaid@umontpellier.fr)

Ce travail sera réalisé exclusivement par :

- CLETZ Laura (laura.cletz@etu.umontpellier.fr)

## Installation

### Prérequis
- Python 3.8+
- Gestionnaire de packages Conda ou pip

### Configuration
```bash
# Installer les dépendances
pip install -r requirements.txt

# Ou avec conda
conda install --file requirements.txt
```

## Utilisation

### Exécuter les scripts Python
```bash
cd python_script/
python svm_script.py
python svm_gui.py
```

### Générer le rapport Quarto

Nous conseillons de produire un fichier HTML plutôt qu'un PDF.

```bash
cd quarto_report/
quarto render svm_report.qmd --to html
# quarto render svm_report.qmd --to pdf
```

## Structure du projet / Project Structure

```
SVM_Practical/
├── README.md
├── LICENSE
├── requirements.txt
├── TP_ML_SVM.pdf
├── .gitignore
│
├── python_script/
│   ├── README.md
│   ├── svm_script.py
│   ├── svm_source.py
│   ├── svm_gui.py
│   └── __pycache__/
│
└── quarto_report/
    ├── README.md
    ├── svm_report.qmd
    ├── svm_report.html
    ├── style.css
    ├── .gitignore
    ├── .quarto/
    └── images/
        ├── BioSSD_logo.png
        ├── FdS.jpg
        ├── UM.png
        ├── linear_C1.png
        ├── linear_C3.png
        ├── linear_C5.png
        ├── poly_degree2.png
        ├── poly_degree3.png
        ├── RBF_C1.png
        └── RBF_C5.png
```
