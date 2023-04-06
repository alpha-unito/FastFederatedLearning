#!/bin/bash -f

MAX_ITER=5
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Master-worker experiments
cd $SCRIPT_DIR/build/examples/masterworker
[ ! -d logs ] && mkdir logs

echo "Running master-worker experiments..."
for WORKERS in 2 4 7
do
	echo "Example with $WORKERS workers..."
	for ITER in $(seq 1 $MAX_ITER)
	do
		echo "Running iteration $ITER..."
		dff_run -V -p TCP -f masterworker_${WORKERS}.json ./masterworker_dist 1 20 5 ../../../data ${WORKERS} 2>&1 > logs/masterworker_${WORKERS}_${ITER}
	done
	echo "Mean execution time for master-worker with $WORKERS workers : " $(cat logs/masterworker_$WORKERS_* | grep "Elapsed time" | awk '{ sum += $3; n++ } END { if (n > 0) print sum / n; }')
done

# Peer-to-peer experiments
cd $SCRIPT_DIR/build/examples/p2p
[ ! -d logs ] && mkdir logs

echo "Running peer-to-peer experiments..."
for WORKERS in 2 4 8
do
	echo "Example with $WORKERS peers..."
	for ITER in $(seq 1 $MAX_ITER)
	do
		echo "Running iteration $ITER..."
		dff_run -V -p TCP -f p2p_${WORKERS}.json ./p2p_dist 1 20 5 ../../../data ${WORKERS} 2>&1 > logs/p2p__${WORKERS}_${ITER}
	done
	echo "Mean execution time for peer-to-peer with $WORKERS peers : " $(cat logs/p2p_$WORKERS_* | grep "Elapsed time" | awk '{ sum += $3; n++ } END { if (n > 0) print sum / n; }')
done

# Edge inference experiments
cd $SCRIPT_DIR/build/examples/edgeinference
[ ! -d logs ] && mkdir logs

echo "Running edge inference experiments..."
for WORKERS in 2 4 7
do
	echo "Example with $WORKERS leaves..."
	for ITER in $(seq 1 $MAX_ITER)
	do
		echo "Running iteration $ITER..."
		dff_run -V -p TCP -f edgeinference_${WORKERS}.json ./edgeinference_dist 1 ${WORKERS} 1 ../../../data/yolov5n.torchscript ../../../data/Ranger_Roll_m.mp4 2>&1 > logs/edgeinference_${WORKERS}_${ITER}
	done
	echo "Mean execution time for edge inference with $WORKERS leaves : " $(cat logs/edgeinference_$WORKERS_* | grep "Elapsed time" | awk '{ sum += $3; n++ } END { if (n > 0) print sum / n; }')
done