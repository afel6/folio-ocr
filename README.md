# folio-ocr

folio-ocr reads English and Arabic company documents into text on your own machines. It
handles born-digital PDFs, scanned PDFs, scans and phone photos. Every line comes with its
box on the page and its review flags. A line read by OCR or by a model gets a confidence score
when the reader gives a usable one. A line nobody could read, or one without a usable score,
has no score and is flagged for review. Lines taken from a PDF's own text layer have no
confidence score; folio checks them with its text-layer validation instead. Export is TXT,
DOCX or JSON.

No OCR tool reads everything correctly, and folio does not pretend to. It flags every line it
cannot vouch for and every region nobody read, so a person checks those. Its accuracy is
measured, never promised.

- Licence: MIT ([LICENSE](LICENSE)). Third-party licences: [THIRD_PARTY.md](THIRD_PARTY.md).
- **Admins** set it up: [Install](#install), [`folio setup`](#download-the-model-files-folio-setup),
  [Tesseract](#optional-tesseract), [model server](#optional-connect-a-language-model),
  [`folio doctor`](#check-the-installation-folio-doctor).
- **Staff** read documents and check flagged lines: [`folio read`](#read-documents-folio-read),
  [review flags](#review-flags), [confidence and review](#confidence-and-review).

Commands marked † have not been tested with this version. They are ordinary Python, Ollama,
Tesseract or Linux package steps. The other commands are folio's own and have been run with
this version.

## What it is and what it is not

It is:

- a command-line tool, `folio`, that reads printed English and Arabic pages line by line;
- offline once set up: reading uses no network unless you connect a model server;
- able to work with three kinds of reader. The built-in OnnxTR reader reads English. Tesseract
  is optional and reads printed Arabic and English. A language-model server is optional and
  must be one you run or choose.

It is not, in this version:

- a model-training tool. It ships no weights for the language model;
- a hosted or multi-user service. It has no accounts and no database;
- a forms or tables tool. It does not extract key-value fields or table structure;
- a reader for languages other than English and Arabic;
- a cloud service. It makes no cloud calls unless you configure a cloud model server;
- an archive. It does not keep documents after the web app stops (see [web app](#web-app-folio-serve));
- a handwriting reader. Handwriting is not measured yet. Its results will be reported, not
  graded pass or fail (see [limitations](#limitations)).

## Install

Requirements:

- Windows or Linux;
- Python 3.12 or 3.13;
- about 145 MB of disk for the model files, plus the Python packages.

Give folio its own virtual environment. Its pinned packages can clash with other tools; for
example, OnnxTR needs `huggingface-hub` below 2.0.

Windows (PowerShell), in the folder that holds `pyproject.toml`:

```powershell
python -m venv .venv                          # †
.venv\Scripts\python.exe -m pip install .     # †
.venv\Scripts\folio.exe --help
```

Linux (bash):

```bash
python3.12 -m venv .venv                      # †
.venv/bin/python -m pip install .             # †
.venv/bin/folio --help                        # †
```

To type plain `folio`, activate the environment first: run `.venv\Scripts\Activate.ps1` in
PowerShell † or `. .venv/bin/activate` in bash †. The examples below write `folio`.

On Linux, the web app (once available) needs an Arabic system font to show Arabic lines,
because it loads no web fonts. Noto Sans Arabic or DejaVu Sans both work. On Debian, the package `fonts-noto-core`
holds Noto Sans Arabic: `sudo apt install fonts-noto-core` †. The package name on other
distributions has not been checked.

## Download the model files: `folio setup`

```powershell
folio setup
```

`folio setup` downloads five files, 144.1 MB in total, into the assets folder:

| File | Size | Used for |
|---|---|---|
| `onnxtr/rep_fast_base-1b89ebf9.onnx` | 42.3 MB | finding text on OCR pages (every language) |
| `onnxtr/crnn_vgg16_bn-743599aa.onnx` | 63.2 MB | reading English words |
| `tessdata/ara.traineddata` | 12.6 MB | Tesseract Arabic (`tessdata_best`) |
| `tessdata/eng.traineddata` | 15.4 MB | Tesseract English (`tessdata_best`) |
| `tessdata/osd.traineddata` | 10.6 MB | Tesseract page orientation (which way is up) |

- Every file is pinned by https URL, size and sha256. A file that does not match is not
  saved. Files already verified are skipped, so running `folio setup` again is safe. Progress
  lines go to stderr.
- The assets folder is `%LOCALAPPDATA%\folio-ocr\assets` on Windows. On Linux it is
  `$XDG_DATA_HOME/folio-ocr/assets`, or `~/.local/share/folio-ocr/assets` when
  `XDG_DATA_HOME` is not set to an absolute path. Set `FOLIO_WEIGHTS_DIR` to use another folder.
- After setup, reading needs no network. A PDF whose pages all keep their text layer needs no
  model files at all.
- The downloads honour the usual proxy variables (`HTTPS_PROXY`). Requests to a model server
  never do; see [privacy](#privacy-and-network).
- For a machine without internet access, run `folio setup` elsewhere and copy the whole assets
  folder across. `folio doctor` then checks every file's sha256.
- The Tesseract files are used only when Tesseract is installed. folio passes them to Tesseract
  itself, so Tesseract's own language packs are not needed.

## Optional: Tesseract

Without Tesseract or a model server, folio cannot read Arabic on scans and photos. With
`--lang ar` or `--lang en+ar`, such lines come back unread and flagged `needs_model_server`.
With Tesseract 5 installed:

- printed Arabic and English lines on scans and photos are read by Tesseract;
- Tesseract's orientation check decides which way is up, in every `--lang` mode. Without it,
  folio decides by a vote of its English recogniser and flags pages it is unsure of.

Install it:

- Windows: install Tesseract 5.4. folio was measured with the UB-Mannheim build
  5.4.0.20240606 †.
- Linux: install Tesseract 5 from your distribution's packages †.

folio looks for the program in this order:

1. `FOLIO_TESSERACT_PATH`, which must be an absolute path to the program;
2. the folders on `PATH`. Only absolute entries count; the current folder never does;
3. on Windows, `%ProgramFiles%\Tesseract-OCR\tesseract.exe`.

Confidence cut-offs are tied to Tesseract's major.minor version. Another version still reads,
but every line it reads needs review until a cut-off exists for that version (see
[confidence and review](#confidence-and-review)).

## Read documents: `folio read`

```text
folio read FILE... [--lang en|ar|en+ar] [--out DIR] [--format txt|docx|json]
```

Input files:

- File types: PDF, PNG, JPEG, TIFF and WebP. folio recognises the type from the file's
  content, not its name.
- Limits: 25 MB per file, 20 pages per PDF or frames per TIFF, and 50 megapixels per page.
  Encrypted PDFs are refused.

**`--lang`** is the language of the pages. The default, `en+ar`, is the safe choice.

- `--lang en` assumes pages without Arabic. Use it only for English-only pages. It reads with
  the built-in OnnxTR reader, with no Tesseract or model server needed.
- `--lang ar` is for Arabic pages, and `--lang en+ar` for pages that mix both languages.

Examples:

```powershell
folio read invoice.pdf                               # text on the screen
folio read scan.jpg --lang en                        # an English-only scan
folio read letter.jpg --out results --format docx    # save results\letter.docx
folio read a.pdf b.png --format json                 # one JSON document per line on stdout
```

After each file, a summary line goes to stderr. These are real runs on folio's test documents:

```text
en-invoice.pdf: 24 lines found, 24 read, 0 need review, 0 unread pages
en-invoice-scan.jpg: 23 lines found, 23 read, 23 need review, 0 unread pages
mixed-letter-scan.jpg: 14 lines found, 0 read, 14 need review, 0 unread pages
hint: use --lang en for English-only pages, or install Tesseract, or set FOLIO_MODEL_URL
```

The summary counts:

- *found*: every line on the pages;
- *read*: lines some reader produced text for;
- *need review*: lines with at least one flag;
- *unread pages*: pages that were not read to the end.

A page with inked areas that no reader read gets its own line after the
summary, for example `scan.png: page 1: 2 areas were not read (see --format json for their
boxes)`.

The born-digital PDF needs no review because its text layer passed every check. On the scan,
every line needs review because no confidence cut-off has been calibrated yet. The mixed
letter was read with the default `--lang en+ar` on a machine with no Tesseract and no model
server, so nobody could read it.

Output formats:

- `txt` (default): the text of each line, one per line. Pages are separated by a form feed.
  An unread line stays as an empty line. A page with areas that were not read starts with a
  note in square brackets saying how many.
- `docx` (needs `--out`): the first paragraph gives the number of lines that need review. Those
  lines are highlighted in yellow, and right-to-left lines are set right to left. A page with
  areas that were not read gets a note under its heading, also highlighted in yellow.
- `json`: the whole document. Every line has its box, text, confidence, flags and status, and
  `unresolved_count` gives the number of lines still needing review plus the number of pages
  with areas that were not read (their boxes are in each page's `unread_regions`).

`--out DIR` creates the folder if it is missing (one level) and names each file after its
input. When that name is one of the input files, or was already written in the same run, folio
adds `-2`, `-3` and so on. Without `--out`, results go to stdout.

Exit codes are 0 when every file was processed, 1 for any error (including a page error), and
130 for Ctrl-C. Lines that need review, or that nobody could read, are not an error. Every error is one line on stderr,
starting `error:` or, for a page error, `<file>: page <n>:`.

## Check the installation: `folio doctor`

`folio doctor` prints one line per check: `ok`, `fail` or `info`. It exits with 1 when any line
is `fail`. Here it is on a Windows machine with Tesseract and no model server:

```text
ok   python: 3.13.1
ok   pdfium: 159.0.8149.0@sourcebuild-toolchained
ok   pdf reader: starts with its limits (1950 MB and 60 s per file)
info assets folder: %LOCALAPPDATA%\folio-ocr\assets
ok   asset tessdata/ara.traineddata: verified (sha256)
ok   asset tessdata/eng.traineddata: verified (sha256)
ok   asset tessdata/osd.traineddata: verified (sha256)
ok   asset onnxtr/rep_fast_base-1b89ebf9.onnx: verified (sha256)
ok   asset onnxtr/crnn_vgg16_bn-743599aa.onnx: verified (sha256)
ok   tesseract: 5.4.0.20240606 at C:\Program Files\Tesseract-OCR\tesseract.exe
info reader onnxtr: uncalibrated (no frozen confidence threshold yet: every line it reads is flagged low_confidence)
ok   reader tesseract: ok
info model server: not configured (set FOLIO_MODEL_URL to use one)
```

The assets folder line shows the real path; it is shortened here. Each `reader` line ends in one
of four states:

- `missing`: the reader's files or program are absent;
- `bad_data`: they are present but cannot be used;
- `uncalibrated`: it reads, but every line it reads needs review;
- `ok`: a confidence cut-off exists for this exact setup.

The PDF and image reader runs in a separate process on Windows and Linux. The 60-second
file-work budget applies to each reader process. Startup has its own time limit, and pages
that go to OCR can start a fresh reader. The budget excludes OCR and model requests.

With a model server configured, doctor adds two more lines. `model self-test` sends two
built-in test lines and passes or fails. `reader model` reports `ok` or `uncalibrated`. The
self-test images are built into folio and are not taken from your documents.

## Optional: connect a language model

folio can send each line image to a model server that speaks the OpenAI chat-completions API
and accepts images. With a server configured, it is used in two places:

- every line on an OCR page is read by the model instead of OnnxTR or Tesseract. OCR pages are
  scans, photos, and PDF pages without a usable text layer;
- on born-digital PDFs, only lines flagged `visual_order` or `text_layer_invalid` are read again
  by the model. Lines a person has edited or confirmed are never sent.

The first read after start, and every `folio doctor` run, sends two built-in test lines: one
English, and one Arabic with digits. If the model gets them wrong, every line it reads is
flagged `model_self_test_failed`.

| Setting | Meaning |
|---|---|
| `FOLIO_MODEL_URL` | The server's OpenAI base URL, usually ending in `/v1`. folio posts to `<URL>/chat/completions`. |
| `FOLIO_MODEL_NAME` | The model name the server expects. Required when `FOLIO_MODEL_URL` is set. |
| `FOLIO_MODEL_API_KEY` | Only for servers that need one. Set it in an environment variable only. It is sent as `Authorization: Bearer`, never logged or printed, and refused over `http://` to another machine. |
| `FOLIO_TWO_READ` | `1` reads each line twice and uses the agreement as its score, for servers that return no log-probabilities. Default `0`. |
| `FOLIO_LINE_TIMEOUT_S` | Time limit per line in seconds (default 60). It also applies to Tesseract. |
| `FOLIO_PAGE_DEADLINE_S` | Time limit per page in seconds (default 900). Lines still waiting at the limit are flagged `not_read`. |

### Ollama (the default example, and the measured setup)

The setup measured for this version:

- Ollama 0.40.16 with the model `qwen3-vl:8b-instruct-q4_K_M` (a 6.2 GB download);
- an RTX 5070 Ti laptop GPU, with `OLLAMA_CONTEXT_LENGTH` unset (Ollama's default context of
  4096);
- peak GPU memory 7.1 GB. The first, cold request took about 53 s, and later lines took about
  1 s each.

These figures come from direct requests to Ollama with folio's prompt on six test lines, not
from a full `folio read` run. The full run is still pending (see [results](#results)).

Use Ollama 0.12.11 or newer. From that version, Ollama returns log-probabilities on its OpenAI
endpoint, and folio scores model lines from them. Without log-probabilities a line has no score
unless `FOLIO_TWO_READ=1`.

1. Install Ollama from ollama.com †. It listens on 127.0.0.1 port 11434 by default, so only
   this machine can reach it.
2. Pull the model †:

   ```powershell
   ollama pull qwen3-vl:8b-instruct-q4_K_M
   ```

3. Optionally, pin the context length by setting `OLLAMA_CONTEXT_LENGTH=4096` in the
   environment of the Ollama server, then restart Ollama †. Ollama's documented default is
   4096, and the context length drives GPU memory.
4. Point folio at it and check:

   ```powershell
   $env:FOLIO_MODEL_URL = "http://127.0.0.1:11434/v1"
   $env:FOLIO_MODEL_NAME = "qwen3-vl:8b-instruct-q4_K_M"
   folio doctor
   ```

   In bash, use `export FOLIO_MODEL_URL=http://127.0.0.1:11434/v1` and
   `export FOLIO_MODEL_NAME=qwen3-vl:8b-instruct-q4_K_M` †.

A local Ollama needs no API key. Its default `127.0.0.1` URL keeps every image on this machine.

### Other OpenAI-compatible servers (not yet tested with folio)

vLLM, the llama.cpp server, LM Studio, NVIDIA NIM or a cloud API can work the same way. Set
`FOLIO_MODEL_URL` to the server's OpenAI base URL and `FOLIO_MODEL_NAME` to its model name.
Reading works, with these limits:

- **Every model line needs review on these servers in this version.** A confidence cut-off is
  tied to the server version and the model digest. folio reads both only the way Ollama
  publishes them (`/api/version` and `/api/tags` next to the `/v1` base URL). On any other
  server, every model line stays flagged `low_confidence`.
- The model must accept images. If the server returns no log-probabilities, set
  `FOLIO_TWO_READ=1`.
- folio connects directly. It uses no proxy for model requests and follows no redirects, so the
  URL must reach the server itself.

> **Privacy warning for cloud and remote servers.** With a model server on another machine,
> and above all a cloud API, an image of every line folio sends leaves this machine. Those
> images are crops of your documents, and the provider's terms decide what happens to them.
> `folio doctor` marks such a server `(leaves this machine)`. Connect one only if your
> company's policy allows these documents to go to that provider. Use an `https://` URL; folio
> refuses to send an API key over `http://` to another machine.

## Privacy and network

- **No network while reading**, except to the model server you configure in
  `FOLIO_MODEL_URL`. Without one, reading is fully offline once `folio setup` has run.
- **What a model server receives**: one grey PNG image per text line, cropped from the page,
  plus a fixed instruction. It never receives the file itself, the file name or any other
  line.
- **No proxy for model requests**: they ignore `HTTP_PROXY` and `HTTPS_PROXY` and follow no
  redirects, so page images cannot leave through a proxy set in the environment. Only
  `folio setup` downloads files, and only that command honours the proxy variables.
- **Leaving the machine**: when the model server's host is not this machine (not a loopback
  address), `folio doctor` marks it `(leaves this machine)`. The web app will report this as
  `leaves_machine` and show a banner.
- **The API key is never logged**, printed or returned. Doctor shows only the model name and
  host, never the URL or the key. Document text is never written to logs.
- **Nothing is written to disk while reading.** folio writes only the output files you ask for.

## Review flags

Every line has a status: `auto_accepted`, `needs_review`, `edited` or `confirmed`. A line with
any flag is `needs_review`. Line flags are in each line's `flags` and page flags in each page's
`flags` in the JSON export. The DOCX export highlights the lines that need review.

Line flags:

| Flag | What it means |
|---|---|
| `low_confidence` | The reader was not sure enough, or no confidence cut-off exists yet for this exact reader setup. |
| `needs_model_server` | Nobody read this line: Arabic or mixed pages with no Tesseract and no model server. |
| `not_read` | Reading this line failed (time limit, server error or Tesseract error). The page error says why. |
| `empty` | The line has no text. |
| `repeated` | The same 20 characters appear 3 or more times, a typical sign of a reader stuck in a loop. |
| `script_mismatch` | The letters do not fit `--lang`: too much Arabic in a `--lang en` line, too much Latin in a `--lang ar` line, or any letter of another script. |
| `presentation_forms` | The line holds Arabic presentation-form characters. They look right but search and copy badly. |
| `lookalike_chars` | The line holds characters that look like normal letters or digits but are not, such as Greek, Cyrillic, Persian letter forms, extended digits or full-width characters. |
| `visual_order` | A born-digital PDF stored this line out of reading order, for example with mirrored brackets or moved `%` and `+`. The text is kept and flagged, never repaired. |
| `text_layer_invalid` | A born-digital PDF's text for this line cannot be trusted: unmapped font codes, private-use or presentation-form characters, control or replacement characters, or an invisible text layer. |
| `invalid_text` | Broken characters were replaced with U+FFFD. |
| `orientation_uncertain` | folio is not sure which way up the page is (also a page flag). |
| `mixed_order` | A model line holds both Arabic and Latin letters. The order of the two runs is not checked, so it always needs review. |
| `truncated` | The model reply was cut off at its length limit. |
| `no_transcription` | The model returned no text. |
| `format_noise` | The model reply holds extra content, such as thinking tags, code fences, line breaks or chatter. The text is kept as sent. |
| `model_self_test_failed` | The model failed the built-in two-line test, so none of its lines can be trusted. |
| `handwritten` | Reserved. This version has no handwriting detector and never sets it. |

Page flags:

| Flag | What it means |
|---|---|
| `unread_regions` | Ink or images outside every line box. Nobody read those areas; their boxes are in the JSON. |
| `invisible_text` | The PDF holds an invisible text layer, typically added by someone else's OCR. Every line on it is flagged `text_layer_invalid`. |
| `empty_with_ink` | The PDF page has ink but no text in its text layer, so folio read it by OCR. This is normal for scanned PDFs. |
| `text_layer_invalid` | More than half of the PDF page's lines failed the text checks, so the page was read by OCR instead. |
| `orientation_uncertain` | folio is not sure which way up the page is. |

In the web app, a reviewer will fix a flagged line by editing it (`edited`) or accept it
unchanged (`confirmed`).

## Confidence and review

In plain words:

1. Every line read on a scan or photo gets a confidence score from 0 to 1. For OnnxTR and
   Tesseract the score is the least certain word in the line. For a model it comes from the
   reply's log-probabilities, or from two reads agreeing. Text from a born-digital PDF has no
   score; folio checks it by rules instead.
2. A line is accepted without review only when its score is at or above a **cut-off (tau)**
   calibrated for that exact setup. The setup is the same reader, model files, version and
   settings. A cut-off is chosen on separate test documents: the lowest value at which the
   missed-error rate stays at or below 10 % with 95 % confidence. It is then graded once on
   documents it has never seen.
3. **Until a cut-off is calibrated for your exact model and server version, every line that
   model reads needs review.** The same holds for OnnxTR and Tesseract. `folio doctor` shows
   `uncalibrated` or `ok` for each reader. A line whose score cannot be computed is flagged
   too.
4. **A model's cut-off holds only for the server version and model digest it was calibrated
   on**, for example Ollama 0.40 and the exact `qwen3-vl:8b-instruct-q4_K_M` build. It also
   depends on folio's fixed prompt and request settings. If you upgrade Ollama to another
   minor version, pull a different build of the model or switch servers, folio finds no
   cut-off, and every model line needs review again.
5. A score at or above the cut-off clears only `low_confidence`. Any other flag, such as
   `mixed_order`, still sends the line to review.

## Web app: `folio serve`

Run `folio serve` and open the address it prints. Choose the document languages and upload a
PDF or image with **Add and read**. Select a flagged line to review it next to the page
image. Edit and **Save** the text or **Confirm** it as shown. Export the document as TXT,
Word (DOCX), or JSON before closing the service.

- It listens on this machine only: `127.0.0.1`, port 8000 (`FOLIO_HOST`, `FOLIO_PORT`).
- **Warning: under `folio serve --allow-remote`, anyone who can reach the port on the network
  can list, read and delete the documents in it.** There are no accounts or passwords.
  `--allow-remote` is required for any `FOLIO_HOST` that is not a loopback address, and the
  server prints a warning when it listens beyond this machine. The default bind reaches this
  machine only.
- It keeps at most 3 documents, in memory only, and forgets them when it stops. Upload limits
  are the same as for `folio read`.
- It shows a banner while the model server is on another machine.
- On Linux it needs an Arabic system font (see [install](#install)).

## Results

The offline missed-error results come from the
[one-time held-out run](eval/results/grade-07ee31ad0cf1-20261120T164035Z.json).
Confidence intervals use resampling of whole documents. The language-model evaluation on the
RTX 5070 Ti laptop is pending.

Character error rate (CER) on printed test documents:

| Documents | Language | Target | Without a model server | With a model server |
|---|---|---|---|---|
| Born-digital PDF | English | ≤ 0.5 % | pending | pending |
| Born-digital PDF | Arabic | ≤ 1 % | pending | pending |
| Clean printed image | English | ≤ 2 % | pending | pending |
| Clean printed image | Arabic | ≤ 3 % | not graded | pending |
| Phone photo, printed | English | ≤ 3 % | pending | pending |
| Phone photo, printed | Arabic | ≤ 5 % | not graded | pending |
| Digits and amounts read exactly | both | ≥ 97 % | pending | pending |

The Arabic image rows are graded with a model server. With a model server, mixed pages count in
the Arabic rows. Without one, their CER is reported only.

The missed-error rate is the share of lines with more than 2 % CER that escaped review. It
includes undetected lines with no unread-region marker. The target is ≤ 10 % as a point
estimate, with the 95 % confidence interval reported beside it.

| Reader setup | Documents | Point estimate | 95 % CI | Bad lines / documents | Good lines flagged | Result |
|---|---|---|---|---|---|---|
| OnnxTR, orientation by Tesseract | English (`--lang en`) | — | — | 121 / 30 | 99.7 % | Not measured (no frozen threshold) |
| OnnxTR, orientation by vote | English (`--lang en`) | — | — | 121 / 30 | 99.7 % | Not measured (no frozen threshold) |
| Tesseract | English (default `--lang en+ar`) | 0.9 % | 0.0–2.8 % | 116 / 31 | 93.7 % | Pass |
| Tesseract | Arabic (default `--lang en+ar`) | 0.1 % | 0.0–0.4 % | 805 / 32 | 97.4 % | Pass |
| Language model (Ollama, qwen3-vl 8B) | English | pending | pending | pending | pending | pending |
| Language model (Ollama, qwen3-vl 8B) | Arabic and mixed | pending | pending | pending | pending | pending |

Tesseract met the missed-error target at a threshold of 0.968 while sending most correct
lines to review. Its one missed error per language was a line the reader did not find.
"Bad lines / documents" counts the bad lines and the documents that contained them.

Coverage, speed and handwriting:

| Measure | Target | Result |
|---|---|---|
| Line recall, clean and scanned pages | ≥ 98 % | pending |
| Line recall, phone photos | ≥ 95 % | pending |
| Model speed, warm, median seconds per page | ≤ 60 s | pending |
| Peak GPU memory with the model | ≤ 11 GB | pending |
| Handwriting, English and Arabic | none: reported, not graded pass or fail | not measured yet |

## Limitations

- **Handwriting is not measured yet.** Its results will be reported, not graded pass or fail.
  There is no handwriting detector, so handwritten lines are not marked as handwritten. Expect
  weak results on them; the confidence rules still apply.
- **Mixed Arabic and English model lines are always flagged** `mixed_order`. folio does not yet
  check the order of the Arabic and Latin runs, and a model can return a mixed line in the
  wrong order.
- **Grids and fax pages may be flagged** `orientation_uncertain`. Without Tesseract, a page with
  no long word, such as a grid of single characters, nearly always is.
- **No tables or forms.** Table cells are read as separate lines, and no table or form
  structure is exported.
- **English and Arabic only.** Letters of other scripts are flagged `script_mismatch`.
- **PDF text is never repaired.** A wrong order or broken characters in a PDF's text layer are
  flagged, not fixed. Rotated Arabic in some PDFs comes out in another order (`visual_order`).
  Text drawn sideways on the displayed page is not detected.
- **Some thin text is missed.** The text detector misses some thin Arabic-Indic digits (such as
  `١`) and some words in some Arabic fonts. What it misses is reported as `unread_regions`.
- **Speed without a model**: an OCR page takes about 4–7 s with OnnxTR and 10–16 s with
  Tesseract on an 8-thread desktop CPU.

## Troubleshooting

| Message | What to do |
|---|---|
| `error: not found: <path>` | Check the file path. |
| `error: --format docx needs --out DIR` | Add `--out <folder>`. |
| `error: <file>: the OCR model files are missing; run folio setup` | Run `folio setup`. |
| `... the OCR model files are damaged; run folio setup` | Run `folio setup`; it downloads the failing file again. |
| `... the OCR model files cannot be read; check the access rights of the assets folder` | Give the user read access to the assets folder. |
| `hint: use --lang en for English-only pages, or install Tesseract, or set FOLIO_MODEL_URL` | Arabic or mixed pages had no reader. Install [Tesseract](#optional-tesseract) or [connect a model](#optional-connect-a-language-model). Use `--lang en` only for English-only pages. |
| `hint: run folio doctor to check the readers` | A page had an error; the line `<file>: page <n>: <reason>` above it says which. Run `folio doctor`. |
| `info tesseract: not found (optional: ...)` | Tesseract is not installed, or not where folio looks; set `FOLIO_TESSERACT_PATH`. |
| `error: FOLIO_TESSERACT_PATH must be an absolute path to the tesseract program` | Use the full path, for example `C:\Program Files\Tesseract-OCR\tesseract.exe`. |
| `error: FOLIO_MODEL_NAME must be set when FOLIO_MODEL_URL is set` | Set `FOLIO_MODEL_NAME`. |
| `fail model self-test: the English test line could not be read: the model server refused the connection; check that it is running` | Start the model server (for Ollama, start the Ollama app or `ollama serve` †), then check the URL and port. |
| `the model server has no such endpoint or model (HTTP 404); check FOLIO_MODEL_URL (usually ending in /v1) and FOLIO_MODEL_NAME` | Fix the URL, or pull the model (`ollama pull <name>` †). |
| `the model server refused the API key or credentials (HTTP 401)` | Check `FOLIO_MODEL_API_KEY`. |
| `the model server did not answer within 60 s (FOLIO_LINE_TIMEOUT_S)` | The first request loads the model and can take close to a minute. Try again, or raise `FOLIO_LINE_TIMEOUT_S`. |
| `the model server is busy (HTTP 429)` | Wait, or ask the provider for a higher limit. |
| `error: FOLIO_MODEL_API_KEY is refused over http:// to a host outside this machine; use an https:// FOLIO_MODEL_URL` | Use an `https://` URL for a remote server. |
| `error: FOLIO_HOST is not a loopback address; folio serve --allow-remote is needed to listen beyond this machine` | Unset `FOLIO_HOST`. Use `--allow-remote` only after reading the [warning](#web-app-folio-serve). |
| Arabic lines show as boxes in the web app (Linux) | Install an Arabic system font (see [install](#install)). |
| `error: unexpected failure (<ExceptionClass>); please report it` | This is a bug. Report the command and the message; never send document content you may not share. |

Model-server messages appear in `folio doctor` and in a read's page lines
(`<file>: page <n>: <message>`). The lines involved are flagged `not_read`.

## Settings

Settings come from environment variables. An empty variable counts as unset. A bad value stops
the command with an error that names the variable, never its value.

| Variable | Default | Meaning |
|---|---|---|
| `FOLIO_MODEL_URL` | unset | Model server base URL (see [above](#optional-connect-a-language-model)). |
| `FOLIO_MODEL_NAME` | unset | Model name; required with `FOLIO_MODEL_URL`. |
| `FOLIO_MODEL_API_KEY` | unset | API key; set it in the environment only. |
| `FOLIO_TWO_READ` | `0` | `1` = read each model line twice when no log-probabilities come back. |
| `FOLIO_LINE_TIMEOUT_S` | `60` | Seconds per line (model and Tesseract). |
| `FOLIO_PAGE_DEADLINE_S` | `900` | Seconds per page with a model server. |
| `FOLIO_TESSERACT_PATH` | unset | Absolute path to the Tesseract program. |
| `FOLIO_WEIGHTS_DIR` | see [`folio setup`](#download-the-model-files-folio-setup) | Assets folder. |
| `FOLIO_HOST` | `127.0.0.1` | Web app address; a non-loopback address needs `--allow-remote`. |
| `FOLIO_PORT` | `8000` | Web app port. |
