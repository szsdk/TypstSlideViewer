# Typst Slide Viewer

## Installation

```bash
pip install .
```

## Usage

```
SYNOPSIS
    gen-slide TYPST_SRC <flags>

POSITIONAL ARGUMENTS
    TYPST_SRC
        Type: str
        Path to the Typst source file.

FLAGS
    -o, --output_file=OUTPUT_FILE
        Type: Optional[]
        Default: None
        Path to the output HTML file.
    -s, --svg_folder=SVG_FOLDER
        Default: 'svgs'
        Folder containing SVG files.
    -t, --template_file=TEMPLATE_FILE
        Type: Optional[]
        Default: None
        Path to the template file.
    -n, --note=NOTE
        Type: Literal
        Default: ''
        Note position. If not specified, assumpe that the slides are not compiled in the speaker mode and the note will be displayed as text in the control window. Only use "right" if the slides are compiled in the speaker mode with `config-common(show-notes-on-second-screen: right)`.
```
