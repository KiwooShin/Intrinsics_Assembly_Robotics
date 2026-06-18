#!/bin/bash
# Run collect_one.sh once per generated config (cfg_0..cfg_{N-1}).
N=${1:-5}
echo "=== collect_set: $N configs | start $(date +%H:%M:%S) ==="
for i in $(seq 0 $((N-1))); do
  CFG=/home/kiwoos/data/configs/cfg_${i}.yaml
  echo ""; echo "############ CONFIG $i  ($(date +%H:%M:%S)) ############"
  bash /home/kiwoos/collect_one.sh "$CFG" "cfg_${i}"
done
echo ""; echo "=== collect_set DONE $(date +%H:%M:%S) ==="
echo "--- results ---"; tail -n $N /home/kiwoos/data/logs/one_log.txt 2>/dev/null
echo "ALLDONE"
