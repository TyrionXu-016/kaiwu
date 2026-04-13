#!/bin/sh
cd /data/projects/robot_vacuum || exit 1
exec sh tools/modelpool_start.sh "$@"
