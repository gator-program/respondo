#!/bin/bash

if [ ! -f scripts/upload_to_anaconda.sh -o ! -f setup.py -o ! -f conda/meta.yaml.in ]; then
	echo "Please run from top dir of repository" >&2
	exit 1
fi
if [ -z "$ANACONDA_TOKEN" ]; then
	echo "Skipping build ... ANACONDA_TOKEN not set." >&2
	exit 1
fi

RESPONDO_VERSION=$(< setup.cfg awk '/current_version/ {print; exit}' | egrep -o "[0-9.]+")
RESPONDO_TAG=$(git tag --points-at $(git rev-parse HEAD))
if [[ "$RESPONDO_TAG" =~ ^v([0-9.]+)$ ]]; then
	LABEL=main
else
	LABEL=dev
	RESPONDO_VERSION="${RESPONDO_VERSION}.dev"
	RESPONDO_TAG=$(git rev-parse HEAD)
fi

echo -e "\n#"
echo "#-- Deploying tag/commit '$RESPONDO_TAG' (version $RESPONDO_VERSION) to label '$LABEL'"
echo -e "#\n"

set -eu

< conda/meta.yaml.in sed "s/@RESPONDO_VERSION@/$RESPONDO_VERSION/g;" > conda/meta.yaml

# Running build and deployment
conda build conda -c adcc/label/main -c defaults --user gator --token $ANACONDA_TOKEN --label $LABEL
