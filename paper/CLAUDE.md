# Paper writing rules

- `paper.tex` is the ICC 2027 (IoT track, EDAS #1571361359) submission in `IEEEtran` conference format: 6 pages plus 1 paid page at most, references included. Rebuild with `make` after every edit and check the page count, overfull boxes and undefined references it prints.
- The bibliography is `paper.bib` (biblatex, `\printbibliography`) and package/macro definitions live in `paper.sty`. Figures live in `figure/`. `slide.tex` and `slide.sty` hold the companion deck; do not copy deck wording into the paper without rewriting it as prose.
- Keep the paper folder clean: build helpers live in `makefile` as targets, never as separate scripts. Generated files (`*.aux`, `*.bbl`, `*.pdf`, ...) are git-ignored.
- Do not edit the preamble macros in `paper.sty` unless asked.
- Use one terminology throughout: "age of incorrect information (AoII)", "irregular repetition slotted ALOHA (IRSA)", "policy" for a transmission rule. Expand every acronym at first use.
- Always format policy names in small caps using `\textsc{...}`: \textsc{Reactive}, \textsc{Throughput}, \textsc{Hybrid}, \textsc{Threshold}, \textsc{UCB} (UCB1 over candidate parameter grid; do not use "Grid-UCB"), \textsc{VHCT} (the baselines) and \tool (ours: defined via \verb|\newcommand{\tool}{\textsc{Mpgts}\xspace}|). Ensure \tool is formatted in small caps everywhere, including algorithm captions and the main text. Introduce \tool in the Introduction using the VScan paper style: `\tool{} (\emph{\underline{M}ultiple-\underline{P}lay \underline{G}aussian \underline{T}hompson \underline{S}ampling})`. Use "we instantiate \tool" (avoid "we propose \tool" in the abstract and introduction to avoid overclaiming).
- Always format dataset names in typewriter font using `\texttt{...}`: \texttt{Berkeley}, \texttt{BTS}, \texttt{SINDIT}, and \texttt{Synthetic}.
- Always use "sensors" for dataset units and device counts (avoid "motes" or inconsistent mixing of "motes" and "streams"; say "sensors" throughout, e.g., in Fig. 1 and main text: 232 sensors for \texttt{Berkeley}, 1633 sensors for \texttt{BTS}, 38 sensors for \texttt{SINDIT}, and 90 sensors for \texttt{Synthetic}).
- Say "proactive retransmission" or "open-loop repetition" for the trigger-conditioned repeats (avoid "blind repair"). Say "retransmission schedule" for $\theta$, "background" (rate $x_{\mathrm{bg}}$) for the rate after the repeats, and "local-record rules" for \textsc{Reactive} and \textsc{Threshold}.
- Avoid the term "collision lockup" or "lockup", as well as "desynchronization" (which is easily confused with PHY/MAC time/slot synchronization in the AoI literature); use "prolonged error period(s)" (matching Munari 2024's $W_k$ error period duration) or "persistent error state(s)".
- Avoid unnatural phrasing like "grant-free Internet of Things (IoT) networks"; attach "grant-free" to the uplink access, channel, or transmission (e.g., "remote sensors transmitting over grant-free wireless uplinks", "grant-free IoT uplinks", or "grant-free random access"). Do not define "Internet of Things (IoT)" in the abstract; IoT is well known, say "remote IoT sensors" directly. Clarify agency: sensors transmit status updates to keep the base station informed, while the base station maintains the remote estimates. Do not introduce parenthetical acronyms in the abstract if they are not reused there (e.g., write "Age of incorrect information" without "(AoII)", and "acknowledgements" without "(ACKs)"); define them at first use in the main text.
- Avoid the term "airtime"; use "transmission cost" or "channel usage".
- For Thompson sampling reward distributions, say "heavy-tailed reward distribution with outlier penalties" (avoid "tight cluster plus spikes").
- Do not use questions or "Why ..." as section or subsection titles (e.g., avoid "Why Local Records Cause Persistent Desynchronization"; use direct descriptive clauses like "Prolonged Error Periods under Local Records").
- Do not use questions or "Why ..." in section summaries or figure callouts/badges (e.g., avoid "Why proactive retransmission matters" in Fig. 1; summarize using direct clauses).
- In the proposed approach section (Section III), do not instantiate concrete hyperparameters (e.g., do not hardcode $|\Theta| = 36$, candidate parameter grids, or specific slot counts in Section III or the algorithm environment). Scan the proposed approach and write it in an abstract, general formulation, leaving all concrete parameter setups to the experimental parameter setup section (Section IV).
- Theorem 1 (Section III-D, "Consistency of the Learned Schedule") is machine-checked in `code/lean/` (`make proof` in `code/`; the theorem-to-file map is `code/lean/README.md`). Keep the paper statement and the Lean hypotheses in sync when either changes, and never mention Lean or the proof files in the paper. The `theorem` and `assumption` environments are IEEEtran `\newtheorem`s, so refer to them with `Theorem~\ref{...}` and `Assumption~\ref{...}` (`\autoref` prints "Sec." for them).
- Every number in Section V comes from `code/result/csv/*.csv` and every table or figure from `make results` in `code/` (`tab-main.tex`, `plot_ablation.pdf`, `plot_scaling_{legend,reward,consensus}.pdf`, `plot_ota.pdf`). Do not type results by hand; rerun and regenerate.
- Say "challenges", never "obstacles". Say "approach", not "path", for methods. Say "formulate X as Y", never "cast X as Y". Avoid the word "sweep" (or "sweeps") for parameter studies; use precise alternatives such as "parameter variations", "operational variations", "sensitivity analysis", or "calibrate across".
- Avoid filler or verbose connectives ("At the same time,", "It is worth noting that"). Use formal wording: write "distinguish" not "tell", and avoid informal or colloquial words such as "sit", "sits", "jump", "jumps", "a lot", "get", "pretty". Use precise technical language instead (e.g., "state transition", "step change", or "abrupt variation" instead of "jump"; "remains", "lies", or "exhibits" instead of "sits").
- Write "e.g." and "i.e." sparingly. Prefer an apposition, a "which" clause, or "so".
- When writing new or revised text, state things directly. Do not use the negative contrast construction "X, not Y" (or "rather than", "instead of") to define a claim; say what it is. Leave existing uses elsewhere alone unless asked.
- Avoid stacking nouns or adding modifiers that the context already implies.
- Do not use em-dashes (`---` or `—`); they can be flagged as irresponsible AI usage. Use en-dashes (`--`) for ranges and rewrite sentences to avoid em-dashes. Never paste Unicode dashes.
- Do not use parentheses in section or subsection titles.
- Cite the latest version of each work (for example the newest arXiv version) and prefer recent citations. Cite or soften every claim that something is a challenge, is important, or is common practice.
- Format LaTeX source with one sentence per line and 2 spaces of indentation for the content of every section and nested environment. Do not add `% ----` section marker comments.
- Refer to figures, tables, sections and algorithms with `\autoref{...}` or `Section~\ref{...}`, and label them `fig:`, `tab:`, `sec:`, `alg:`.
- Do not leave `\tl{...}` notes or `TODO` text in compiled output. Keep unresolved items in `TODO.md`.

# Content and style (distilled from advisor comments in the lab's papers)

## Introduction and framing
- Introduce our work early: "We propose ..." appears within the first few paragraphs (avoid "In this paper, we..."). Summarize prior work in a few sentences focused on the limitation we address; details belong in Related Work.
- Do not include a paper organization paragraph at the end of the Introduction; it is redundant in a conference paper.
- Keep problem paragraphs focused on background, challenges and limitations of prior work. Place our metric definitions and findings in the contributions list or technical sections.
- Define every technical term in plain words at first use, and write for a reader outside the subfield.
- Avoid artificial announcements such as "leave two gaps. First... Second...". State the problems directly.
- Do not cite forward to later sections in the Introduction; keep its narrative self-contained.
- Make contributions concrete: the method is one contribution, the dataset-driven evaluation another. Claim "first" only when no prior work defined the same thing, and present standard practice as standard, with citations.

## Technical sections
- Open every section and subsection with a sentence on its purpose: whether it is our contribution or background, and why the reader needs it.
- Follow formal steps with an "Intuitively, ..." explanation in plain terms, and justify every design choice so that nothing looks like magic or a cherry-picked example.
- Define every symbol before use; remove notation and concepts that are never used later.
- When a point was made earlier, refer back instead of restating it; delete sentences that repeat earlier ones.

## Evaluation
- Open the Evaluation with the research questions, then describe the setup, baselines (reactive, random, hybrid) and metrics (AoII, utilization, reward) that answer them.
- Justify every number and setting (device count, channels, seeds, horizon, hyperparameters): if it affects results, say how; if it does not, remove it, since unexplained settings read as cherry-picking.
- Report what affects fairness and reproducibility: dataset sources and preprocessing, the simulator version, hardware, and the number of runs.
- Make every number traceable to a table, figure, or computation. Expand the main result; do not spend text defending small gains or disparaging baselines.
- Do not raise limitations that invite questions we cannot answer unless we address them.
- Keep the Evaluation from dominating the paper, and avoid walls of dense text.

## Conclusion
- Give concluding remarks, insights, implications and future directions, citing the work that inspires them. Do not re-summarize the results.

## Presentation
- Introduce and explain every figure and table in the text, and place it near the text that discusses it.
- Keep captions short: 2 lines on average, 3 at most. State only what the figure shows; setup and how to read it belong in the prose that cites it.
- In the compiled PDF, the last line of a paragraph must contain at least three words. Citations such as "[3]" or "[2]--[4]" do not count; an inline equation counts as one word. Rephrase (or tie words with `~`) and rebuild to check.
- Split long paragraphs into normal paragraphs (not `\paragraph{}` headings), and split long sentences.
- Avoid generic or meaningless sentences; be specific about which dataset, which policy, which metric.

# Final check protocol (triggered when user says "final check")
- Inspect all paragraphs so that none has two or fewer words on its final line in the compiled PDF.
- Fit the paper within the page limit, with the last reference near the end of the final page, and equalize the two columns on the last page.
- Verify that `make` compiles with 0 errors, 0 overfull boxes and 0 undefined references.
