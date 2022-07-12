#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Run this script at project root by ".linter.sh" before you commit.


{
  black --version | grep -E "21\." > /dev/null
} || {
  echo "Linter requires 'black==21.*' !"
  exit 1
}

ISORT_VERSION=$(isort --version-number)
if [[ "$ISORT_VERSION" != 4.3* ]]; then
  echo "Linter requires isort==4.3.21 !"
  exit 1
fi

set -v

echo "Running isort ..."
isort -y -sp . --atomic

echo "Running black ..."
black -l 100 .

echo "Running flake8 ..."
if [ -x "$(command -v flake8)" ]; then
  flake8 .
else
  python3 -m flake8 .
fi

command -v arc > /dev/null && arc lint
