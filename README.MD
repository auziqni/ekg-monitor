# EKG MONITOR

## creating app

create main.py file
create README.md file
create .gitignore file

### git init

```bash
git init
git add .
git commit -m "initial commit"

```

### create virtual environment

```bash
python -m venv env
```

### activate virtual environment

```bash
source env/Scripts/activate
```

### install dependencies

```bash
pip install pyqtgraph PyQt5 pyserial numpy
```

### create requirements.txt

```bash
pip freeze > requirements.txt
```

### deactivate virtual environment

```bash
deactivate
```

## clone app

instll from requirements.txt

```bash
pip install -r requirements.txt
```
