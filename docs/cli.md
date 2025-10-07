# Command-Line Interface

FairML Datasets provides a command-line interface (CLI) for common operations on fairness datasets. This page documents the available commands and their options.

## Overview

The CLI is accessible via the `fairground` module:

```bash
python -m fairground [COMMAND] [OPTIONS]
```

## Available Commands

::: mkdocs-click
    :module: fairground.__main__
    :command: cli
    :prog_name: python -m fairground
    :depth: 1

## Examples

### Generating Metadata

Generate and save metadata for all datasets:

```bash
python -m fairground metadata
```

Export metadata in JSON format:

```bash
python -m fairground metadata -f metadata.json
```

Generate metadata for a specific dataset:

```bash
python -m fairground metadata --id adult
```

### Exporting Datasets

Export all datasets in prepared format:

```bash
python -m fairground export-datasets --stage prepared
```

Export a specific dataset with train/test/validation splits:

```bash
python -m fairground export-datasets --id adult --stage split
```

Include usage information:

```bash
python -m fairground export-datasets --include-usage-info
```

### Exporting Citations

Export citations for all datasets:

```bash
python -m fairground export-citations
```

Export citations for specific datasets:

```bash
python -m fairground export-citations --ids adult,compas
```