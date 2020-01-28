# Project
Spring 2020 CS 289 HW1

# Author
Takuma Kinoshita (3034401803)

# Instructions
Each type of output will be saved at the following directories:
- Log: `log/`
- Figures: `figures/`
- Submission File: `data/output/`

Usage of the main script `src/run.py`:
```
python src/run.py [-h] problem data_name
```
- `problem`: problem number (e.g. 2, 3, 4)
- `data_name`: dataset name (e.g. mnist, spam, cifar10)

## 2. Data Partitioning

```
python src/run.py 2 mnist
python src/run.py 2 spam
python src/run.py 2 cifar10
```

## 3. Support Vector Machines: Coding

```
python src/run.py 3 mnist
python src/run.py 3 spam
python src/run.py 3 cifar10
```

## 4. Hyperparameter Tuning
```
python src/run.py 4 mnist
```

## 5. K-Fold Cross-Validation
```
python src/run.py 5 spam
```

## 6. Kaggle
```
python src/run.py 6 mnist
python src/run.py 6 spam
python src/run.py 6 cifar10
```