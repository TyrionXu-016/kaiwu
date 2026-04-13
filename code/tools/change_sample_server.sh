#!/bin/sh
# train_test runs with cwd=/workspace/code; real scripts expect /data/projects/robot_vacuum.
cd /data/projects/robot_vacuum || exit 1
exec sh tools/change_sample_server.sh "$@"
