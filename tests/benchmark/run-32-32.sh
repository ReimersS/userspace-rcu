#!/usr/bin/env bash
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Run each benchmark with 32 readers / 32 writers for DURATION seconds.
# Each benchmark gets WARMUP discarded runs followed by MEASURED reported runs.
# Usage: run-32-32.sh [DURATION] [WARMUP] [MEASURED]

if [ "x${URCU_TESTS_SRCDIR:-}" != "x" ]; then
	UTILSSH="$URCU_TESTS_SRCDIR/utils/utils.sh"
else
	UTILSSH="$(dirname "$0")/../utils/utils.sh"
fi

SH_TAP=1
source "$UTILSSH"

DURATION=${1:-10}
WARMUP=${2:-5}
MEASURED=${3:-1}
TIMEOUT=$((DURATION * 2))

TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"; _exit' EXIT

TEST_ARRAY="test_urcu_gc test_urcu_mb_gc test_urcu_qsbr_gc
            test_urcu_lgc test_urcu_mb_lgc test_urcu_qsbr_lgc
            test_urcu test_urcu_mb test_urcu_qsbr
            test_rwlock test_perthreadlock test_mutex"

NUM_TESTS=$(( 12 * MEASURED ))
plan_tests ${NUM_TESTS}

for TEST in ${TEST_ARRAY}; do
	for _w in $(seq 1 "${WARMUP}"); do
		diag "warmup ${_w}/${WARMUP}: ${TEST}"
		timeout "${TIMEOUT}" "${URCU_TESTS_BUILDDIR}/benchmark/${TEST}" 32 32 "${DURATION}" >/dev/null 2>&1
	done
	for _m in $(seq 1 "${MEASURED}"); do
		okx ${URCU_TESTS_TIME_BIN} timeout "${TIMEOUT}" "${URCU_TESTS_BUILDDIR}/benchmark/${TEST}" 32 32 "${DURATION}" 2>"${TMPFILE}"
		diag "time: $(cat "${TMPFILE}")"
	done
done
