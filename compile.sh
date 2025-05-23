#!/bin/zsh

c_modules=(
    "decoder" \
)

GREEN="\x1B[32m"
BOLD="\033[1m"
EOC="\x1B[0m"

for module in $c_modules; do
	echo "$GREEN$BOLD-> compile $module$EOC"
    cd phyling/$module
    python3 setup.py build_ext --inplace
    cd -
done

exit 0
