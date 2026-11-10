# Validation

Version: 2.0.0

## Completed checks

- 34 automated test methods passed, including parameterized cases.
- The synthetic demo ran directly from the project folder and from another working directory.
- The command-line help and single-task/four-stage workbook outputs were checked.
- All code cells in the 22 notebooks compiled without undefined module-level names.
- Date assignment checks covered nonconsecutive and duplicate dataframe indices.
- Binary missing-value assignments returned the expected labels.
- External date evaluation ran with synthetic workbooks, including missing predictions.
- An installable wheel built and imported from an isolated target directory, including its prompts.

## Run the checks

```bash
python demo.py
python run_pipeline.py --help
python -m unittest discover -s tests -v
python -m compileall -q smmnlp tests demo.py run_pipeline.py
```

The external notebook execution test requires the notebook plotting dependency.
It is skipped if matplotlib is not installed.

## Scope

Model adapter tests use doubles. These checks establish application behavior with
synthetic inputs; they do not measure clinical accuracy, real checkpoint prediction
parity, or GPU throughput. Clinical notebook runs require their named study files,
reference annotations, human-review workbooks, and a working local model environment.
