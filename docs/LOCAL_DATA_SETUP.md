# Competition data setup

The match bundle is distributed by the organisers to enrolled participants (see [`DATA.md`](../DATA.md)); it is not redistributed here.

Keep the competition files separate from the committed repository. Use the project scripts and tools to process and evaluate them.

## Minimal dataset to begin (tracking only)

From the bundle's \`tactical\` folder, take:

- \`BAR-ATH_tactical_atd.txt\`
- \`BAR-ATH_tactical_atd.xml\`

These two files suffice to regenerate the official baseline without the video.

For subsequent geometry and segmentation experiments, take also:

- \`BAR-ATH_tactical_smart_tagging.xml\`
- \`BAR-ATH_tactical_homography.csv\`

No need to download the 5.2 GB tactical video or 14 GB broadcast material until we specifically decide to work with visual features.

## Where to put the files on a Mac

Open Terminal:

~~~bash
mkdir -p "$HOME/mtah2027/tactical"
~~~

Move the **four original downloaded files** into that \`tactical\` folder using Finder. Do not rename them. The required result is:

~~~text
~/mtah2027/
└── tactical/
    ├── BAR-ATH_tactical_atd.txt
    ├── BAR-ATH_tactical_atd.xml
    ├── BAR-ATH_tactical_smart_tagging.xml
    └── BAR-ATH_tactical_homography.csv
~~~

If you later download the video, place \`BAR-ATH_tactical.mp4\` next to those files. Optional broadcast files go under \`~/mtah2027/broadcast/\`.

Keep the data folder **outside** the Git repository, and **outside** any shared/public directory.

## Working commands

~~~bash
cd "$HOME/bcn-gradient-depression"   # replace with the path where you cloned this repo
source .venv/bin/activate              # if the environment is already created
export MTAH_DATA_DIR="$HOME/mtah2027"
python scripts/check_local_data.py --level features
python src/predict.py --window 50 --output submissions/draft_v0.csv
python evaluation/validate_submission.py submissions/draft_v0.csv
~~~

If you haven't cloned it yet:

~~~bash
git clone https://github.com/aleetreny/bcn-gradient-depression.git
cd bcn-gradient-depression
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
export MTAH_DATA_DIR="$HOME/mtah2027"
python scripts/check_local_data.py --level tracking
python src/predict.py --window 50 --output submissions/draft_v0.csv
~~~

**Note:** \`src/predict.py\` currently runs the official baseline with temporal smoothing. It is not a new improved model. Its output is intentionally ignored by git.

## Security / reproducibility division

- **GitHub (private, later public for submission):** source code, pinned dependencies, algorithm description, public organiser references, tests and reproduction commands.
- **Competition assets:** videos, ATD tracking, Smart Tagging, homography and derived granular observations stay outside committed source code.
- **Experiments:** version the scripts, dependencies and configurations, while keeping licensed source data out of public Git history.
- **Official Google Form:** the authorised channel for final submission, including the competition CSV (according to their terms).

Keep source code versioned, experiments reproducible and match assets separate from Git commits.

## Important technical detail

The \`tactical\` source defines the 199,319 output frames. The \`broadcast\` source is **not** synchronised and must not accidentally be used as the submission timeline. The official \`games.py\` handles frame IDs and half-dependent offsets.

We should begin with ATD and Smart Tagging before investing in video decoding or hosted GPU infrastructure.
