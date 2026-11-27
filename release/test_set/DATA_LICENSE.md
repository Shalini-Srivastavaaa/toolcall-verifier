# License of the published test set (`release/test_set/`)

The test set is a mix of material with different owners, so it carries **two licenses plus a third for one part**. Use it under all that apply to what you use.

| Part of the data | Who owns it | License |
|---|---|---|
| **Our human labels and annotations**: `label` (correct / incorrect), `error_class` and the class names, `origin` (natural / induced), `class7_reason` (sub-reasons of Class 7) and the label scheme and labeling guide in the README | the authors of this repository | **Creative Commons Attribution 4.0 International (CC BY 4.0)**, text in [`licenses/CC-BY-4.0.txt`](licenses/CC-BY-4.0.txt), summary at https://creativecommons.org/licenses/by/4.0/ |
| **tau2-bench content**: the conversation text (user, agent and tool-result turns), the tool calls, the tool schemas and the domain policy text, of the 363 real test rows and of the τ²-bench-based rows of `test_synth` | Sierra Research | **MIT License**, "Copyright (c) 2025 Sierra Research"; the notice is reproduced below and in [`licenses/tau2-bench-MIT.txt`](licenses/tau2-bench-MIT.txt) |
| **BFCL content**: the function-calling conversations and function schemas of the 198 `bfcl` rows of `test_synth` | UC Berkeley Gorilla project and contributors | **Apache License 2.0**, text in [`licenses/BFCL-Gorilla-Apache-2.0.txt`](licenses/BFCL-Gorilla-Apache-2.0.txt); these rows are modified by us (see below) |
| **Our modifications** in `test_synth` (synthetic corruptions of correct calls, and the paired controls) | the authors | CC BY 4.0, on top of the license of the material they modify |

## Attribution

If you use the labels, please cite the repository and the writeup (Shalini Srivastava, *toolcall-verifier*, 2026, https://github.com/Shalini-Srivastavaaa/toolcall-verifier) and say the labels are "CC BY 4.0, toolcall-verifier human labels". If you use the conversation text you must also keep the Sierra Research MIT notice below, and for the `bfcl` rows the Apache-2.0 license and a statement that the rows were modified.

## τ²-bench MIT notice (must accompany copies of the τ²-bench content)

> MIT License
>
> Copyright (c) 2025 Sierra Research
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Things this license file cannot settle

- **Model-generated text.** The conversations in τ²-bench's recorded simulations were produced by language models run by the benchmark's authors (agents such as gpt-4.1, gpt-4.1-mini, o4-mini and claude-3-7-sonnet, and a simulated user). Sierra's MIT license covers what Sierra can license; the terms of the model providers for that output were reviewed by the project owner: outputs are owned by the user under the OpenAI and Anthropic terms, and the data is not used to build competing models (`docs/licenses.md` 3.2; the owner's reading, not legal advice). Anyone reusing the text is responsible for checking those terms for their own use.
- **Personal data.** The conversations are simulated customers from τ²-bench's own databases. The check we ran on the release is in `release/test_set/README.md` ("Personal data check"); it cannot prove the absence of real personal data, only that nothing outside τ²-bench's own records was found by the patterns used.
- **No warranty.** The labels come from human review under a written guide, with written rules applied to some of them, and some are debatable (see the README of the test set); the data is provided as is.
