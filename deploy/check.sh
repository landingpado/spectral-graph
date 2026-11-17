#!/usr/bin/env bash
# Check that a running explorer serves both pages, the viewer data and every API endpoint.
#
#   bash deploy/check.sh                                   the server on this machine, port 8080
#   bash deploy/check.sh https://explorer.yourdomain.com   the public address (no trailing slash)
#
# Prints one line per check and exits non-zero if any failed. Each check is tried twice: on a freshly
# started server the first query can hit the 5 s limit while the database is still being read from disk.
BASE="$1"
if [ -z "$BASE" ]; then BASE="http://127.0.0.1:8080"; fi
failed=0

check() {   # label, path, text the response must contain
  for attempt in 1 2; do
    body=$(curl -fsS --max-time 30 "$BASE$2" 2>&1)
    if [ $? -eq 0 ] && printf '%s' "$body" | grep -q -- "$3"; then
      if [ "$attempt" = 1 ]; then echo "ok    $1"; else echo "ok    $1  (second try)"; fi
      return
    fi
  done
  echo "FAIL  $1   $BASE$2"
  printf '      %s\n' "$(printf '%s' "$body" | head -c 160)"
  failed=$((failed + 1))
}

check "3-D view page"                 "/"                                              "js/app.js"
check "year view page"                "/years.html"                                    "js/years.js"
check "viewer code"                   "/js/app.js"                                     "loadStore"
check "vendored three.js"             "/vendor/three.module.js"                        "WebGLRenderer"
check "data manifest"                 "/data/manifest.json"                            '"tracks"'
check "case index"                    "/data/cases.json"                               "Miranda v. Arizona"

chunk=$(curl -fsS --max-time 30 "$BASE/data/manifest.json" 2>/dev/null | grep -o 'pos_[0-9]*_[0-9]*\.bin' | head -1)
size=$(curl -fsS --max-time 60 -o /dev/null -w '%{size_download}' "$BASE/data/$chunk" 2>/dev/null)
if [ -n "$chunk" ] && [ "${size:-0}" -gt 100000 ]; then echo "ok    position chunk $chunk  ($size bytes)"
else echo "FAIL  position chunk '$chunk'  (got ${size:-0} bytes)"; failed=$((failed + 1)); fi

check "API: court list"               "/api/courts"                                    '"scotus"'
check "API: cases filed in a year"    "/api/cases?year=1973&court=scotus&q=roe"        "Roe v. Wade"
check "API: one decision"             "/api/case/108713"                               "Blackmun"
check "API: who cites it"             "/api/cites/108713?direction=in&limit=5"         "case_name"
check "API: bench on a date"          "/api/bench?court=scotus&date=1973-01-22"        "Blackmun"
check "API: timeline"                 "/api/timeline?court=scotus&start=1970&end=1975" '"year":1973'

echo
if [ "$failed" -eq 0 ]; then echo "all checks passed for $BASE"; else echo "$failed check(s) failed for $BASE"; fi
exit "$failed"
