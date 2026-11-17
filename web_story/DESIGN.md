---
name: Pizarra táctica del vestuario
description: The FC Barcelona possession-challenge story, explained by the coach in chalk on a dressing-room tactics board.
colors:
  slate: "#1C2722"
  slate-raised: "#222E29"
  tray: "#141B18"
  frame: "#5A635E"
  chalk: "#ECEEE6"
  chalk-body: "rgba(236, 238, 230, 0.86)"
  chalk-soft: "rgba(236, 238, 230, 0.64)"
  chalk-line: "rgba(236, 238, 230, 0.16)"
  yellow-chalk: "#EECC5C"
typography:
  display:
    fontFamily: "Kalam, \"Comic Sans MS\", cursive"
    fontSize: "clamp(1.9rem, 2.55vw, 2.7rem)"
    fontWeight: 400
    lineHeight: 1.14
  headline:
    fontFamily: "Kalam, \"Comic Sans MS\", cursive"
    fontSize: "clamp(1.6rem, 2.6vw, 2.3rem)"
    fontWeight: 400
    lineHeight: 1.15
  title:
    fontFamily: "Kalam, \"Comic Sans MS\", cursive"
    fontSize: "1.32rem"
    fontWeight: 400
    lineHeight: 1.2
    fontFeature: "\"tnum\""
  body:
    fontFamily: "Barlow, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "clamp(0.98rem, 1.06vw, 1.06rem)"
    fontWeight: 400
    lineHeight: 1.62
  body-small:
    fontFamily: "Barlow, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "0.86rem"
    fontWeight: 400
    lineHeight: 1.3
  label:
    fontFamily: "\"Barlow Condensed\", \"Arial Narrow\", sans-serif"
    fontSize: "1rem"
    fontWeight: 600
    lineHeight: 1
rounded:
  sm: "4px"
  md: "6px"
  full: "50%"
spacing:
  rail: "56px"
  gutter: "clamp(24px, 4vw, 64px)"
  sheet-pad: "clamp(22px, 3vw, 40px)"
  control-gap: "10px"
components:
  button-chalk:
    backgroundColor: "transparent"
    textColor: "{colors.chalk}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: "9px 16px 8px"
  button-chalk-primary:
    backgroundColor: "transparent"
    textColor: "{colors.chalk}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: "9px 16px 8px"
  rail-magnet:
    backgroundColor: "#2B3530"
    textColor: "{colors.chalk-soft}"
    typography: "{typography.label}"
    rounded: "{rounded.full}"
    size: "40px"
  rail-magnet-active:
    backgroundColor: "{colors.chalk}"
    textColor: "{colors.tray}"
    rounded: "{rounded.full}"
    size: "40px"
  chalk-tray-rail:
    backgroundColor: "{colors.tray}"
    textColor: "{colors.chalk}"
    height: "{spacing.rail}"
  summary-sheet:
    backgroundColor: "{colors.slate-raised}"
    textColor: "{colors.chalk}"
    rounded: "{rounded.md}"
    padding: "{spacing.sheet-pad}"
  sheet-tab:
    backgroundColor: "transparent"
    textColor: "{colors.chalk-soft}"
    padding: "10px 0 8px"
  sheet-tab-active:
    textColor: "{colors.chalk}"
---

# Design System: Pizarra táctica del vestuario

## Overview

**Creative North Star: "The Coach's Board at Half-Time"**

The whole surface is one green-black slate in a dressing room. The pitch is printed on it, thin and permanent; everything the coach adds is chalk, grainy and hand-wobbled, written stroke by stroke as the reader scrolls and wiped with a felt eraser before the next chapter. Notes sit straight on the slate to the left of the pitch, the running score is chalked under it, and the chapter magnets are stuck along the aluminium chalk tray at the top. Nothing floats, glows or tilts in 3D: depth exists only where a physical object on a real board has it (the tray, the magnets, the eraser).

The palette is three inks and nothing else: the slate, white chalk, and one stick of yellow chalk reserved for the ball and for precision. Type splits by who is "speaking": the coach's hand (Kalam) for anything written on the board, Barlow for the reading prose, Barlow Condensed for the hardware of the tray. Density is calm and one-thing-at-a-time: one chapter is on the board at once, and the previous one survives only as the faint ghost a wiped board keeps.

Confirmed rejections: the dark neon 3D "tech flight", the white data-explainer page, glowing tech looks, many colours, and generic AI palettes (cream + serif + terracotta, near-black + neon).

**Key Characteristics:**
- One canvas board (fixed, full viewport) with a DOM notes column and rail over it.
- Three inks: slate, white chalk, one yellow chalk.
- Chalk grain masks the hand-lettered DOM text so notes and drawing are the same material.
- Draw-on and erase are the motion grammar; no fades-for-decoration, no parallax, no glow.
- Barça drawn as O, Athletic as X, both in white chalk; the ball always yellow.

## Colors

A near-monochrome green-black slate with white chalk at graded strengths and a single warm yellow accent.

### Primary
- **Yellow Chalk** (`yellow-chalk`): the only accent. On the board it draws the ball and its dotted path, the interception burst and "robo" segments, the newest running-score figure, the double circle around the final figure and the "+27.6 pts" under it. In the DOM it marks the summary sheet's figure badges, the yellow underline of the primary finale button, the focus ring, and (at 32% alpha) text selection. A second chalk stick in the tray echoes it.

### Neutral
- **Dressing-Room Slate** (`slate`): page and canvas background; the board itself. Also the 94% translucent backing of the notes card on phones.
- **Raised Slate** (`slate-raised`): the summary sheet's surface, one step lighter than the board.
- **Chalk-Tray Black** (`tray`): the top rail, scrollbar track, and the text colour on the active white magnet.
- **Aluminium Frame** (`frame`): the 4px lip under the rail and the 3px frame around the summary sheet.
- **White Chalk** (`chalk`): headlines, figures, active states, Barça O and Athletic X, every chalk stroke on the board.
- **Chalk Body** (`chalk-body`): reading prose in the notes and the sheet; already-visited magnets.
- **Soft Chalk** (`chalk-soft`): captions under figures, idle magnets, idle tabs.
- **Chalk Line** (`chalk-line`): the hairline under the sheet's tab row. Other dividers are dashed chalk at 20-28% alpha; button outlines are chalk at 34-40% alpha, rising to 60-70% on hover.

### Named Rules
**The One Yellow Stick Rule.** Yellow means the ball or precision, nothing else. A new element earns yellow only if it is the ball, a figure of the precision climb, or the focus ring. Decoration, headings and states of navigation stay white.

**The White Magnet Rule.** The active chapter magnet turns white chalk (`chalk` face, `tray` numerals), never yellow. Navigation state is white; yellow is reserved for the story's numbers.

**The Three Inks Rule.** No fourth hue. Team identity is shape (O versus X), not colour; Barça and Athletic are both white chalk.

## Typography

**Display Font:** Kalam 400 (with "Comic Sans MS", cursive), self-hosted
**Body Font:** Barlow 400 / 600 (with "Helvetica Neue", Arial, sans-serif), self-hosted
**Label Font:** Barlow Condensed 500 / 600 (with "Arial Narrow", sans-serif), self-hosted

**Character:** A coach's marker hand set against a sturdy, sporty grotesque. Kalam is the voice on the board; Barlow is the printed handout you read beside it; Barlow Condensed is the stamped hardware of the tray.

### Hierarchy
- **Display** (Kalam 400, clamp(1.9rem, 2.55vw, 2.7rem), 1.14, balanced wrap): chapter headline in the notes column, written word by word.
- **Headline** (Kalam 400, clamp(1.6rem, 2.6vw, 2.3rem), 1.15): the summary sheet title.
- **Title** (Kalam 400, 1.32rem, 1.2, tabular numerals): key figures under a chapter's prose and summary card headings.
- **Body** (Barlow 400, clamp(0.98rem, 1.06vw, 1.06rem), 1.62, max 64ch): chapter prose; emphasis is Barlow 600 in full `chalk`. Summary cards use 0.95rem / 1.6.
- **Body small** (Barlow 400, 0.86rem, 1.3, `chalk-soft`): captions paired with a Kalam figure.
- **Label** (Barlow Condensed 600, 0.8-1.08rem, no uppercase): rail brand, magnet numerals, buttons, "Cerrar", figure badges. Tabs use 500 at rest, 600 when active.
- **On-canvas chalk** (Kalam, sized in pitch metres, never below 12px): pitch annotations, chapter title on the board, the running score (16-34px, scaled to the tally slot) and its "precisión" label.

### Named Rules
**The Who-Is-Speaking Rule.** If the coach would write it on the board, it is Kalam and carries the chalk-grain mask. If you read it, it is Barlow. If it is part of the board's hardware, it is Barlow Condensed. Never set prose in Kalam or a headline in Barlow.

**The Chalk Grain Rule.** Every Kalam element in the DOM is masked with the same grain board.js publishes as `--chalk-grain` (96px tile), so DOM lettering and canvas chalk read as one material.

## Layout

A fixed full-viewport canvas is the board; the page scrolls a tall invisible track that drives the drawing. Over it sit two fixed layers: the chalk-tray rail (`rail`, 56px) across the top and the notes layer.

- **Laptop and up (860px+):** notes column on the left at the `gutter` inset, width min(36vw, 520px), starting clamp(28px, 6vh, 64px) below the rail. The pitch fills the remaining width to the right (scaled to fit 116 x 80 pitch metres), with the board's chapter title above it and the running score chalked beneath it, spanning the pitch's width.
- **Rail:** brand left, eight magnets centred (7px gaps), "Ver resumen" at the right end. Below 1100px the chalk sticks in the tray are dropped; below 900px the brand is dropped and the magnets centre; below 520px magnets shrink to 32px with 3px gaps.
- **Phones (<860px):** the pitch spans the full width with 16px side padding, still landscape; the notes become a translucent slate card (12px inset, `rounded.md`, chalk hairline border) that steps aside after the reading window. A short-height variant (<=760px tall) tightens notes type further.
- **Summary sheet:** centred dialog, min(1080px, 95vw), max 88vh, cards in an auto-fit grid (min 270px columns, 36px column gap) separated by dashed chalk rules rather than boxes.
- **Rhythm:** spacing is set in ems inside text blocks (0.55-1.5em) and as clamps tied to viewport at the layout level; there is no global numeric scale beyond the tokens above.

**Open item (not a rule):** phones keep the landscape pitch. A portrait pitch was considered and deferred; future work may revisit it, and nothing here should be read as prohibiting it.

## Elevation & Depth

The board is flat. Depth appears only on physical objects that would cast it on a real dressing-room board: the aluminium tray, the magnets stuck to it, the felt eraser, and the summary sheet lifted off the board. Chalk itself never has a shadow or glow.

### Shadow Vocabulary
- **Tray ledge** (`box-shadow: inset 0 -7px 0 #0E1311, 0 6px 16px rgba(0, 0, 0, 0.4)`): the rail's inner lip and the soft shadow it casts on the slate.
- **Magnet** (`box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 2px 3px rgba(0, 0, 0, 0.55)`): a round magnet sitting proud of the tray.
- **Sheet lift** (`box-shadow: 0 24px 60px rgba(0, 0, 0, 0.55)`): the summary sheet over a 78% slate-black scrim.
- **Eraser** (canvas shadow, offset 5px / 12px, blur 18px, black at 50%): the felt eraser held against the board during the wipe.

### Named Rules
**The Real Objects Only Rule.** A shadow is allowed only on something you could pick up off a real tactics board. Text, chalk strokes, the pitch and the notes column stay flat.

## Shapes

Mostly rectilinear with gently softened corners (`rounded.md`, 6px) on buttons, the sheet and the phone notes card; the focus ring rounds at `rounded.sm` (4px). Magnets are perfect circles (`rounded.full`). Borders are thin chalk outlines (1px at 34-40% alpha) or the heavier aluminium frame (3-4px `frame`). Dividers inside content are dashed, like a chalk line, not solid rules. On the canvas, every coach-drawn shape is a hand-wobbled polyline drawn in two passes (full width at 95% alpha, then 55% width at 55% alpha offset 0.7px) under a grain mask; printed pitch lines are clean 1.4px strokes at 30% alpha.

## Components

### Buttons
Chalk-outlined, quiet, and hardware-lettered.
- **Shape:** gently softened (`rounded.md`).
- **Default (`button-chalk`):** transparent, `chalk` Barlow Condensed 600, 1px chalk outline at 40%; rail variant "Ver resumen" is slightly smaller (0.95rem, 6px 14px 5px, outline at 34%).
- **Primary (`button-chalk-primary`):** same, outline at 70%, and a 2px `yellow-chalk` underline offset 5px; used for the finale call to action.
- **Hover / Focus:** background fills with chalk at 8-10%, outline rises to 60-70%, 0.2s ease. Focus is the global 2px yellow ring, offset 3px.

### Navigation: Chalk-Tray Rail and Magnets
- **Rail:** `tray` ground, 4px `frame` lip, tray-ledge shadow; two chalk sticks (white, then yellow) lie in the tray on wide screens.
- **Magnets:** 40px round, face #2B3530, Barlow Condensed 600 numerals of each chapter's figure (or "Regla" / "Final"). Idle `chalk-soft`; visited `chalk-body`; hover lifts 1px and brightens to `chalk`; active is a white chalk face with `tray` numerals and `aria-current="step"`. An invisible touch area extends 7px above and below.

### Summary Sheet
- **Surface:** `slate-raised`, 3px `frame` border, `rounded.md`, sheet-lift shadow, rises 16px and fades in (0.45s, ease-out curve).
- **Tabs:** Barlow Condensed text tabs on a `chalk-line` hairline; active tab gets full `chalk`, weight 600 and a 2px white chalk underline (never yellow).
- **Cards:** no boxes; Kalam heading, then a yellow Barlow Condensed figure badge, then Barlow body, separated by dashed chalk rules.

### Notes Column (signature)
Chapter text written straight on the slate. The Kalam headline is written word by word; prose settles in; key figures sit under a dashed chalk rule as Kalam numerals with Barlow captions. When the eraser crosses into the next chapter, the notes are wiped by the same eraser edge as the drawing (mask driven by board.js's eraser position).

### The Board (signature)
Printed pitch, 22 tokens (Barça O, Athletic X), the yellow ball, chapter diagrams drawn stroke by stroke with scroll, the running score crossed out and rewritten under the pitch until the final figure is circled twice, and a felt eraser that wipes left to right in passes, leaving each chapter's blurred ghost behind.

## Do's and Don'ts

### Do:
- **Do** keep every new element inside the three inks: `slate` family, `chalk` at its graded strengths, and `yellow-chalk` only for the ball, precision figures and focus.
- **Do** set anything the coach writes in Kalam with the `--chalk-grain` mask, and reading prose in Barlow at 1.6+ line height within 64ch.
- **Do** introduce new chapters by drawing them on and leave them by wiping with the eraser; respect `prefers-reduced-motion` by showing the finished drawing and notes without the write-in.
- **Do** distinguish teams by mark (O and X), not by colour.
- **Do** use dashed chalk rules to separate content and outline buttons in chalk at 34-40%.

### Don't:
- **Don't** add neon, glow, gradients-as-decoration or 3D camera moves; this world refuses the dark "tech flight".
- **Don't** turn the surface white or paper-like; the white data-explainer page is the other refused default.
- **Don't** use yellow for navigation state, headings or decoration; the active magnet and active tab are white chalk.
- **Don't** shadow chalk, text or the pitch; shadows belong only to the tray, magnets, eraser and the lifted sheet.
- **Don't** add club crests, logos or official colours for either team.
