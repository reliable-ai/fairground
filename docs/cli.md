# Command-Line Interface

FairML Datasets provides a command-line interface (CLI) for common operations on fairness datasets. This page documents the available commands and their options.

## Overview

The CLI is accessible via the `fairml_datasets` module:

```bash
python -m fairml_datasets [COMMAND] [OPTIONS]
```

## Available Commands

::: mkdocs-click
    :module: fairml_datasets.__main__
    :command: cli
    :prog_name: python -m fairml_datasets
    :depth: 1

## Examples

### Generating Metadata

Generate and save metadata for all datasets:

```bash
python -m fairml_datasets metadata
```

Export metadata in JSON format:

```bash
python -m fairml_datasets metadata -f metadata.json
```

Generate metadata for a specific dataset:

```bash
python -m fairml_datasets metadata --id adult
```

### Exporting Datasets

Export all datasets in prepared format:

```bash
python -m fairml_datasets export-datasets --stage prepared
```

Export a specific dataset with train/test/validation splits:

```bash
python -m fairml_datasets export-datasets --id adult --stage split
```

Include usage information:

```bash
python -m fairml_datasets export-datasets --include-usage-info
```

### Exporting Citations

Export citations for all datasets:

```bash
python -m fairml_datasets export-citations
```

Export citations for specific datasets:

```bash
python -m fairml_datasets export-citations --ids adult,compas
```