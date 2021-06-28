# Global Homework 1

## Environment preparation

```sh
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Script running

`letters.py` is the relevant script file. Executed as `python3 letters.py {inputfile}`, outputs `output.html` with the letters, `output.txt` with image names belonging to the appropriate clusters.

## Extras

- `output.j2` a Jinja2 template, necessary for generating the HTML file, is provided;
- `letters.ipynb` with the code, in which are also comments;
- If above doesn't run (or one doesn't care about actually running the notebook), a frozen version `letters.pdf` is provided;
- Input files for training dataset (`train` directory with images and `train.txt`), along with results (`output.html` and `output.txt`) are provided

## Description of the method

It's provided in the `letters.pdf`, I won't therefore repeat it here (not to mention there are requisite plots in there). As for the execution time, it runs in <20s on my laptop (which in itself is not very beefy)

## "How to run"

I already spoke of it but apparently it's a required section so whatever.

```sh
# Prepare the environment
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
# Run the script
python3 letters.py {inputfile}
```
