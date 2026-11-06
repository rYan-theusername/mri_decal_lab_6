# MRI DeCal — Lab 6: Writing Pulse Sequences

Student materials for Lab 6 of the MRI DeCal. In this assignment you write real pulse sequences in Python with PyPulseq: an FID, a spin echo that rephases the spins from Lab 5, and a 2D gradient echo image. You then run them on a simulated object with MRZero, and use them to explore spoiling and image contrast.

This repository does **not** include solutions.

## Requirements

- Python 3.12 or newer (`python3 --version`)
- A way to run Jupyter notebooks, such as [Cursor](https://cursor.com), [VS Code](https://code.visualstudio.com/) with the Jupyter extension, JupyterLab, or Jupyter Notebook

## Setup

1. Clone this repository and enter the folder:

```bash
git clone https://github.com/rYan-theusername/mri_decal_lab_6.git
cd mri_decal_lab_6
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install the Python packages. MRZero installs PyTorch, so this step can take a few minutes:

```bash
pip install -r requirements.txt
```

4. Open `lab_6.ipynb` and run the first cell. If it prints `All libraries are installed.`, you are ready to start.

If you prefer the Jupyter web interface:

```bash
pip install notebook
jupyter notebook lab_6.ipynb
```

## Working on the lab

- Fill in every `# TODO` in `lab_6.ipynb`.
- Short written answers go in the cyan *Your answer here.* cells.
- Helper functions live in `helpers.py` and `visualizations.py`. You do not need to edit those files.
- Keep `helpers.py` and `visualizations.py` in the same folder as the notebook.
- The first simulation of the brain downloads MRZero's brain phantom, so you need an internet connection the first time you run Part 3.
- Some cells render animations. The first render of a cell can take a few seconds.

## Files

| File | What it is |
| --- | --- |
| `lab_6.ipynb` | The assignment notebook |
| `helpers.py` | Helpers for sequence timing, building a gradient echo, and simulating with MRZero |
| `visualizations.py` | Sequence diagrams, signal plots, and animation helpers |

If you have questions about the lab, reach out to the course staff.
