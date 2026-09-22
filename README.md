# Typst Slide Viewer

## Installation

```bash
npm install
```

This project requires Node.js 22.15 or newer and the `typst` executable on your
`PATH`. Install the command globally if you want to use it outside this checkout:

```bash
npm link
```

The browser-side Zstandard and TAR libraries are normal npm dependencies; the
generator embeds their browser bundles into each output HTML file.

## Usage

After installation, the `gen-slide` command is available in the terminal (or use
`node bin/gen-slide.js` directly from this checkout).

```text
SYNOPSIS
    gen-slide TYPST_SRC <flags>

POSITIONAL ARGUMENTS
    TYPST_SRC
        Path to the Typst source file.

FLAGS
    -o, --output_file=OUTPUT_FILE
        Path to the output HTML file.
    -s, --svg_folder=SVG_FOLDER
        Folder containing SVG files.
    -t, --template_file=TEMPLATE_FILE
        Path to the template file.
    -n, --note=NOTE
        Note position. If not specified, assumpe that the slides are not compiled in the speaker mode and the note will be displayed as text in the control window. Only use "right" if the slides are compiled in the speaker mode with `config-common(show-notes-on-second-screen: right)`.
```

### Example

```bash
cd examples/
wget 'https://download.samplelib.com/mp4/sample-5s.mp4'
gen-slide slides.typ
```

Then an HTML file, `slides.html`, will be generated in the same directory.
Open it with a browser to view the slides.

Press `w` (or use the control-window toolbar button) to open the presenter controls.
The control window can record microphone audio, play the completed recording, and
download it in the audio format supported by your browser. Your browser will ask
for microphone permission when recording starts.

### HTML embeds

Write the bundled Typst helper into the current directory:

```bash
gen-slide html-embed
```

Choose another destination, or replace an existing copy:

```bash
gen-slide html-embed --output path/to/html-embed.typ
gen-slide html-embed --force
```

### HTML placeholders

Generate fallback screenshots for literal `html-embed` targets with the bundled
browser-based command. Chromium, Chrome, Edge, Brave, or Firefox must be
available on your `PATH`.

```bash
gen-slide placeholders slides.typ
```

Keep these placeholder images up to date when embedded HTML or video content
changes. Thumbnail generation uses them during Typst's PNG render, and changes
to their contents automatically invalidate the WebP thumbnail cache.
