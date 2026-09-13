#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
rebar3 compile
exec escript scripts/life.escript "$@"
