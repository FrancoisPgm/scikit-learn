#!/bin/bash

set -e
set -x

cd ../../

python -m venv test_env
source test_env/bin/activate

python -m pip install scikit-learn/scikit-learn/dist/*.tar.gz
python -m pip install pytest pandas

# Run the tests on the installed source distribution
mkdir $TEST_DIR
cd $TEST_DIR

pytest --pyargs sklearn --junitxml=$JUNITXML -o junit_family=legacy
