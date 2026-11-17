# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The jury and organisers of the FC Barcelona possession challenge. They know football and data science, and they open the link alone on a laptop to walk through the team's solution at their own pace before or after the defence.

## Product Purpose

A scroll-driven story that explains how the team took an automatic possession labeller from 61.2% to 88.8% on the challenge metric, one step at a time, so the jury understands why each step worked and trusts the result. Success: a juror can retell the five steps and the final numbers after one pass.

## Positioning

It is told through the match itself, the way a coach would: each chapter draws on the Barça vs Athletic Club pitch the specific failure (blind nearest-player rule, the 70% continuous-overlap rule, camera glitches, throw-ins, 0.88 s interceptions) and the fix that removed it, while the running score is crossed out and rewritten from 61.2% to 88.8%. No other entry can show this exact climb on this exact match.

## Operating Context

- Delivered alongside the hackathon submission (ENTREGA_FINAL_HACKATHON); viewed privately behind a password on Cloudflare Pages (`bcn-web-story.pages.dev`).
- Static site in `web_story/`: plain HTML/CSS/JS, a canvas chalkboard drawn live by `board.js`, Lenis for inertial scrolling, self-hosted fonts in `fonts/`. Deployed with `deploy-cloudflare.sh`.
- Language: Spanish.

## Capabilities and Constraints

- Story text and every figure are fixed and must not change: 61.2% → 73.1% → 80.0% → 81.3% → 86.9% → 88.8% (+27.6 pts), 197 possessions, 61.5% / 38.5%, 199,319 frames, 7 of 7 matches improved, +7.1 pts in a new stadium, 0 false alarms.
- Eight chapters, three summary tabs; jury can jump between chapters.
- Must stay private (no public hosting), work on laptop first and still read on phones.
- No real match footage or club crests are available or allowed; visuals are synthetic renders of the tracking data story.

## Brand Commitments

- FC Barcelona and Athletic Club are named as the teams of the match; no club logos or official branding are used.
- The user rejected: neon / glowing "tech" looks, many colours, generic AI-looking design (cream + serif + terracotta, near-black + neon).

## Evidence on Hand

- All metrics and narrative in `web_story/index.html` (COPY array, summary dialog).
- Synthetic chalk diagrams of each step, authored in `board.js` from the method's facts (formations are illustrative, not tracking data).
- No photos, testimonials or external press exist; none may be invented.

## Product Principles

- Every visual must explain a step of the method, not decorate it.
- The numbers are the plot: the climb from 61.2% to 88.8% must be felt, not just read.
- Football first, machine learning second: the jury should recognise the game before the model.
- Honest: what was tried and discarded stays in the story.
