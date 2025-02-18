#!/bin/bash

PID_FILE="/tmp/literag_crawler.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Stopping crawler (PID: $PID)..."
    kill $PID
    rm "$PID_FILE"
    echo "Crawler stopped"
else
    echo "Crawler is not running"
fi 