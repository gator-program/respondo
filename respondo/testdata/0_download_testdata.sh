#!/bin/bash

# taken from adcc repository, written by @mfherbst

SOURCE="https://q-chem.de/respondo_testdata"
DATAFILES=(
	data_0.0.1.tar.gz
)
#
# -----
#

THISDIR=$(dirname "${BASH_SOURCE[0]}")
cd "$THISDIR"
echo "Updating testdata ... please wait."

download() {
	if which wget &> /dev/null; then
	    echo $@
		wget -w 1 -qN --show-progress $@
	else
		echo "wget not installed" >&2
		exit 1
	fi
}

download $(for file in ${DATAFILES[@]}; do echo $SOURCE/$file; done)

if which sha256sum &> /dev/null; then
	sha256sum --ignore-missing -c SHA256SUMS || exit 1
fi

for file in ${DATAFILES[@]}; do
    tar -xzf $file
done

exit 0
