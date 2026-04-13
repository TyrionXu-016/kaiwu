#!/bin/sh
cd /data/projects/robot_vacuum || exit 1
exec sh tools/stop.sh "$@"
