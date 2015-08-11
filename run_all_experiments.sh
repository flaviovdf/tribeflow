#!/bin/bash

NUM_THREADS=3
NUM_TOPIC_INITIAL=100
NUM_ITER=2000
NUM_BATCHES=10
HOSTS=localhost

#mu0 v0 sigma0
PRIORS_NRM_LOC=(7200 1 720) #location data
PRIORS_NRM_LFM_YOO=(300 1 30) #other data

OUT_FOLDER=models-same-count
mkdir $OUT_FOLDER 2> /dev/null

IN_FOLDER=traces/small

#1. Run the trace with the NormalKernel
#for f in $IN_FOLDER/*.dat; do
#    bname=`basename $f`
#    out_file=$OUT_FOLDER/$bname-nrm-dyn.h5
#    
#    if [ "$bname" == "brightkite.dat" ] || [ "$bname" == "four_sq.dat" ]; then
#        mpiexec --host $HOSTS -np $NUM_THREADS python main.py $f $NUM_TOPIC_INITIAL $out_file \
#            --num_iter $NUM_ITER --num_batches $NUM_BATCHES \
#            --kernel tstudent \
#            --residency_priors "${PRIORS_NRM_LOC[@]/#/+}" --dynamic True \
#            --leaveout 0.3
#    else
#        mpiexec --host $HOSTS -np $NUM_THREADS python main.py $f $NUM_TOPIC_INITIAL $out_file \
#            --num_iter $NUM_ITER --num_batches $NUM_BATCHES \
#            --kernel tstudent \
#            --residency_priors "${PRIORS_NRM_LFM_YOO[@]/#/+}" --dynamic True \
#            --leaveout 0.3
#    fi
#done

#2. Run the trace with no Kernel
for f in $IN_FOLDER/*.dat; do
    out_file=$OUT_FOLDER/`basename $f`-noop-not-dyn.h5
    mpiexec --host $HOSTS -np $NUM_THREADS python main.py $f $NUM_TOPIC_INITIAL $out_file \
        --num_iter $NUM_ITER --num_batches $NUM_BATCHES \
        --kernel noop --leaveout 0.3
done

#2. Run the trace witn the NormalKernel and not Dynamic
#for f in $IN_FOLDER/*.dat; do
#    bname=`basename $f`
#    out_file=$OUT_FOLDER/$bname-nrm-not-dyn.h5
#    if [ "$bname" == "brightkite.dat" ] || [ "$bname" == "four_sq.dat" ]; then
#        mpiexec --host $HOSTS -np $NUM_THREADS python main.py $f $NUM_TOPIC_INITIAL $out_file \
#            --num_iter $NUM_ITER --num_batches $NUM_BATCHES \
#            --kernel tstudent \
#            --residency_priors "${PRIORS_NRM_LOC[@]/#/+}" --leaveout 0.3
#    else
#        mpiexec --host $HOSTS -np $NUM_THREADS python main.py $f $NUM_TOPIC_INITIAL $out_file \
#            --num_iter $NUM_ITER --num_batches $NUM_BATCHES \
#            --kernel tstudent \
#            --residency_priors "${PRIORS_NRM_LFM_YOO[@]/#/+}" --leaveout 0.3
#    fi
#done

#3. Run the trace with the Bernoulli Kernel
for f in $IN_FOLDER/*.dat; do
    out_file=$OUT_FOLDER/`basename $f`-ber-dyn.h5
    mpiexec --host $HOSTS -np $NUM_THREADS python main.py $f $NUM_TOPIC_INITIAL $out_file \
        --num_iter $NUM_ITER --num_batches $NUM_BATCHES \
        --kernel eccdf \
        --residency_priors 1 $(($NUM_TOPIC_INITIAL - 1)) --dynamic True \
        --leaveout 0.3
done

#4. Run the trace with the Bernoulli Kernel and not Dynamic
#for f in $IN_FOLDER/*.dat; do
#    out_file=$OUT_FOLDER/`basename $f`-ber-not-dyn.h5
#    mpiexec --host $HOSTS -np $NUM_THREADS python main.py $f $NUM_TOPIC_INITIAL $out_file \
#        --num_iter $NUM_ITER --num_batches $NUM_BATCHES \
#        --kernel eccdf \
#        --residency_priors 1 $(($NUM_TOPIC_INITIAL - 1)) --leaveout 0.3
#done
