#!/bin/sh
cd /data/projects/robot_vacuum || exit 1
exec sh tools/change_sample_server.sh "$@"
