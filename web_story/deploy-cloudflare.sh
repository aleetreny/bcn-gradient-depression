#!/usr/bin/env bash
# Deploy the committed web_story to Cloudflare Pages (project bcn-web-story), behind the
# password lock in functions/_middleware.js. Only files committed at HEAD are uploaded, so
# local scratch files never leave the machine. The password lives in Cloudflare, not here:
#   npx wrangler pages secret put SITE_PASSWORD --project-name bcn-web-story
# Without it the site answers 503 to everyone.
set -euo pipefail

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
rev=$(git -C "$repo" rev-parse --short HEAD)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git -C "$repo" archive HEAD web_story | tar -x -C "$tmp"
mv "$tmp/web_story/functions" "$tmp/functions"
mkdir "$tmp/public"
mv "$tmp/web_story/index.html" "$tmp/web_story/"*.js "$tmp/web_story/.nojekyll" "$tmp/public/"
for d in assets fonts; do
  if [ -d "$tmp/web_story/$d" ]; then mv "$tmp/web_story/$d" "$tmp/public/"; fi
done

cd "$tmp"
npx -y wrangler@4 pages deploy public --project-name bcn-web-story --branch main \
  --commit-hash "$rev" --commit-message "web_story $rev" --commit-dirty=true
