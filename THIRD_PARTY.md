# Third-party software and data

folio-ocr is released under the MIT licence (see `LICENSE`). It installs, downloads or keeps
in its repository the third-party parts listed below, each under its own licence.

Each licence below was checked against the package's PyPI metadata and the licence files in
its wheel, the GitHub licence of the source repository, or the licence file and name table of
each font.

## 1. Python packages installed by `pip install .`

### Pinned in `pyproject.toml`

| Package | Version | Licence | Notes |
|---|---|---|---|
| pillow | 13.0.0 | MIT-CMU | Image reading. The wheel bundles image libraries; their notices ship inside it. |
| pypdfium2 | 5.15.2 | BSD-3-Clause and Apache-2.0, plus the licences of the bundled PDFium dependencies | PDF text and rendering. The dependency licences ship in the wheel (`licenses/BUILD_LICENSES`). |
| rapidfuzz | 3.14.6 | MIT | Edit distances. |
| httpx | 0.28.1 | BSD-3-Clause | Downloads for `folio setup`. Model-server requests do not use it. |
| python-docx | 1.2.0 | MIT | DOCX export. |
| onnxtr[cpu-headless] | 0.9.0 | Apache-2.0 | Text detection and English word recognition. The metadata says "Apache License"; the LICENSE file and the repository say Apache-2.0. |
| onnxruntime | 1.32.0 | MIT | Runs the OnnxTR model files. Bundled parts are listed in `ThirdPartyNotices.txt` in the wheel. |
| opencv-python-headless | 5.1.0.95 | Apache-2.0; bundled FFmpeg under LGPL-2.1 | Image operations for OnnxTR. The wheel ships FFmpeg as a separate library (`cv2/opencv_videoio_ffmpeg*.dll` on Windows); see `cv2/LICENSE-3RD-PARTY.txt`. |
| numpy | 2.5.5 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Arrays. |
| fastapi | 0.165.0 | MIT | Web app (`folio serve`). |
| uvicorn | 0.54.0 | BSD-3-Clause | Web server for `folio serve`. |
| python-multipart | 0.0.34 | Apache-2.0 | File upload in the web app. |
| starlette | 1.10.0 ‡ | BSD-3-Clause | Web app; imported directly as well as through fastapi. |
| pydantic | 2.14.4 ‡ | MIT | Web app request checks; imported directly as well as through fastapi. |

‡ The pin for this package is being added to `pyproject.toml`. The version shown is the one
installed with fastapi 0.165.0.

### Installed with them

These packages were installed on Windows with Python 3.13 (checked 2026-11-20). folio does not
pin them, so other platforms or later installs may get different versions.

| Package | Version | Licence | Pulled in by |
|---|---|---|---|
| anyio | 4.16.0 | MIT | httpx |
| certifi | 2026.10.23 | MPL-2.0 | httpx |
| httpcore | 1.0.9 | BSD-3-Clause | httpx |
| h11 | 0.16.0 | MIT | httpcore |
| idna | 3.20 | BSD-3-Clause | anyio |
| typing_extensions | 4.16.0 | PSF-2.0 | anyio |
| lxml | 6.1.3 | BSD-3-Clause | python-docx |
| scipy | 1.18.1 | BSD-3-Clause (bundled parts listed in the wheel) | onnxtr |
| pyclipper | 1.4.0 | MIT | onnxtr |
| langdetect | 1.0.9 | Apache-2.0. Its metadata says MIT, but the LICENSE file in the wheel is Apache-2.0. | onnxtr |
| six | 1.17.0 | MIT | langdetect |
| huggingface_hub | 1.33.0 | Apache-2.0 | onnxtr. folio does not download through it: `folio setup` fetches the pinned URLs listed in `src/folio_ocr/assets.py`. |
| click | 8.5.1 | BSD-3-Clause | huggingface_hub |
| filelock | 4.2.12 | MIT | huggingface_hub |
| fsspec | 2026.9.0 | BSD-3-Clause | huggingface_hub |
| hf-xet | 1.8.0 | Apache-2.0 | huggingface_hub |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | huggingface_hub |
| PyYAML | 6.0.3 | MIT | huggingface_hub |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | huggingface_hub |
| colorama | 0.4.6 | BSD-3-Clause (LICENSE file; Windows only) | tqdm |
| defusedxml | 0.7.1 | PSF-2.0 | onnxtr |
| anyascii | 0.3.3 | ISC | onnxtr |
| flatbuffers | 25.12.19 | Apache-2.0 | onnxruntime |
| protobuf | 7.36.2 | BSD-3-Clause | onnxruntime |
| pydantic-core | 2.50.5 | MIT | pydantic |
| annotated-types | 0.8.0 | MIT | pydantic |
| typing-inspection | 0.4.5 | MIT | pydantic, fastapi |
| annotated-doc | 0.0.7 | MIT | fastapi |
| opentelemetry-api | 1.45.2 | Apache-2.0 | fastapi |

certifi and tqdm (MPL-2.0) and the FFmpeg library inside opencv-python-headless (LGPL-2.1)
carry weak-copyleft licences. folio uses all three unmodified, as installed packages.

### Development only (`pip install ".[dev]"`; folio does not need them to run)

| Package | Version | Licence |
|---|---|---|
| pytest | 9.2.0 | MIT |
| pytest-cov | 7.1.0 | MIT |
| jiwer | 4.0.0 | Apache-2.0 |
| fpdf2 | 2.9.0 | LGPL-3.0-only |
| uharfbuzz | 0.57.0 | Apache-2.0 |
| fonttools | 4.68.0 | MIT |
| ruff | 0.17.5 | MIT |

The packages these pull in are not listed.

## 2. Files downloaded by `folio setup`

`folio setup` downloads these files once into the assets folder. It checks each file's size
and sha256 against the pinned values and does not save a file that fails the check. Reading
never downloads anything. Exact URLs and sha256 values are in `src/folio_ocr/assets.py`.

| File | Size (bytes) | Source | Licence |
|---|---|---|---|
| `tessdata/ara.traineddata` | 12,603,724 | Tesseract `tessdata_best` repository, commit `e12c65a9` | Apache-2.0 |
| `tessdata/eng.traineddata` | 15,400,601 | Tesseract `tessdata_best` repository, commit `e12c65a9` | Apache-2.0 |
| `tessdata/osd.traineddata` | 10,562,727 | Tesseract `tessdata` repository, commit `13665561` (the file Tesseract 5.4.0 ships) | Apache-2.0 |
| `onnxtr/rep_fast_base-1b89ebf9.onnx` (text detector, fast_base) | 42,342,927 | OnnxTR GitHub release v0.0.1 | Apache-2.0 |
| `onnxtr/crnn_vgg16_bn-743599aa.onnx` (word recogniser) | 63,213,845 | OnnxTR GitHub release v0.7.1 | Apache-2.0 |

The total is 144,123,824 bytes (144.1 MB). The `tessdata_best` and `tessdata` licences are
the GitHub licences of the `tesseract-ocr/tessdata_best` and `tesseract-ocr/tessdata`
repositories. The OnnxTR model files carry the Apache-2.0 licence of the OnnxTR repository that
publishes them, and `assets.py` records the same licence for each file.

## 3. Software you install yourself (not shipped)

- **Language model.** folio ships no model weights for the language model and downloads none.
  If you connect a model server, you install the server and the model yourself, under their
  own licences. The default example uses **Ollama**, whose GitHub repository is MIT-licensed,
  and **Qwen3-VL**, whose GitHub repository is Apache-2.0-licensed. Check the licence of the
  exact model build you pull.
- **Tesseract** (optional). You install the program yourself. Its repository is Apache-2.0.
  folio runs it as a separate program and gives it the language data listed in section 2.

## 4. Files kept in this repository (not installed by `pip install .`)

### Fonts

All nine font files are unmodified and licensed under the SIL Open Font License 1.1. Each one
has its licence file next to it, and its name table points to the OFL. They are used only to
generate test and evaluation documents.

| Font file | Folder | Licence file | Copyright holder (from the licence file) |
|---|---|---|---|
| Amiri-Regular.ttf | `tests/fixtures/fonts/` | OFL-Amiri.txt | The Amiri Project Authors |
| NotoNaskhArabic-Regular.ttf | `tests/fixtures/fonts/` | OFL-NotoNaskhArabic.txt | The Noto Project Authors |
| NotoSans-Regular.ttf | `tests/fixtures/fonts/` | OFL-NotoSans.txt | The Noto Project Authors |
| CourierPrime-Regular.ttf | `tests/dev_fonts/` | OFL-CourierPrime.txt | The Courier Prime Project Authors |
| CrimsonText-Regular.ttf | `tests/dev_fonts/` | OFL-CrimsonText.txt | The Crimson Text Project Authors |
| IBMPlexSansArabic-Regular.ttf | `tests/dev_fonts/` | OFL-IBMPlexSansArabic.txt | IBM Corp. (Reserved Font Name "Plex") |
| Lateef-Regular.ttf | `tests/dev_fonts/` | OFL-Lateef.txt | SIL Global; Latin glyphs The Crimson Pro Project Authors (Reserved Font Names "Lateef", "SIL") |
| PT_Sans-Web-Regular.ttf | `tests/dev_fonts/` | OFL-PTSans.txt | ParaType Ltd. (Reserved Font Names "PT Sans", "ParaType") |
| ScheherazadeNew-Regular.ttf | `tests/dev_fonts/` | OFL-ScheherazadeNew.txt | SIL Global (Reserved Font Names "Scheherazade", "SIL") |

The test PDFs and images in `tests/fixtures/` were made with these fonts. So were the two
built-in model self-test images, which are PNG bytes inside
`src/folio_ocr/model_self_test_lines.py` and contain no font data. The OFL states that its
requirement for fonts to stay under the licence "does not apply to any document created
using the Font Software".

### Unicode test data

`tests/data/unicode/15.0.0/` and `tests/data/unicode/15.1.0/` hold
`bidi_character_test_sample.txt`, `bidi_test_sample.txt` and
`derived_bidi_class_unassigned.txt`. These are samples of `BidiCharacterTest.txt`,
`BidiTest.txt` and `DerivedBidiClass.txt` from the Unicode Character Database. They are © Unicode, Inc., under the Unicode terms of use
(<https://www.unicode.org/terms_of_use.html>). Each file keeps its original copyright line and
names those terms in its header. The full Unicode licence text is not yet stored next to them.
