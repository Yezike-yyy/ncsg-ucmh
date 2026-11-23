#!/usr/bin/env bash
# Fetch official third-party baseline repositories at the exact audited commits.
# These repositories are fetched directly from their original authors and are not
# redistributed in this repository.
set -euo pipefail

mkdir -p external

if [ ! -d "upstream" ]; then
  git clone https://github.com/penghu-cs/UCCH.git upstream
  git -C upstream checkout 0c20e62b99875cd2ec9d7a496eae80b3ab8ba61b
fi

if [ ! -d "external/djsrh-upstream" ]; then
  git clone https://github.com/zs-zhong/DJSRH.git external/djsrh-upstream
  git -C external/djsrh-upstream checkout 6041c2df810723dd0052e2e5b7c6bd33033f0f21
fi

if [ ! -d "external/jdsh-upstream" ]; then
  git clone https://github.com/KaiserLew/JDSH.git external/jdsh-upstream
  git -C external/jdsh-upstream checkout caa8224e2f2e028042fe270cab318c3b96590e41
fi

echo "All upstream repositories checked out at exact audited commits."
