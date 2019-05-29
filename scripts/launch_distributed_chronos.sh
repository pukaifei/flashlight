#!/bin/bash
function usage()
{
    echo "Usage: $0 [-n <NUM_NODES>][-p <NUM_PROCESSES_PER_NODE>][-f <COMMAND_FILE> | <COMMAND>]"
}

while getopts n:p:f: parm ; do
case $parm in
  n)
    nnodes=$OPTARG
    ;;
  p)
    nproc_per_node=$OPTARG
    ;;
  f)
    cmd_file=$OPTARG
    ;;
  *)
    usage
    echo "Invalid argument"
esac
done

shift $((OPTIND-1))

# calling directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if  [[ -z $cmd_file ]]; then
  training_script=$*
else
  if [[ ! -f "$cmd_file" ]]; then
    usage
    echo "File not found : $cmd_file"
    exit 1
  fi
  training_script=$(cat "$cmd_file")
fi

world_size=$(($nnodes * $nproc_per_node))

hostgroup=fblearner_ash_bigbasin_fair
rndvpath="/mnt/vol/gfsai-east/ai-group/users/${USER}/chronos/rendezvous"
solibdir="/mnt/vol/gfsai-east/ai-group/users/${USER}"
cpu_per_proc=6
cpu_per_node=$((nproc_per_node*cpu_per_proc))

if [ "$nnodes" -eq 1 ] && [ "$nproc_per_node" -eq 1 ]
then
  jobid=$(echo "AF_PATH=${solibdir} ${training_script}" \
      | /usr/local/chronos/scripts/crun --hostgroup "${hostgroup}" --gpu "${nproc_per_node}" --cpu "${cpu_per_node}" --secure-group oncall_fair_speech --timeout 29d)
elif [ "$nnodes" -eq 1 ]
then
  jobid=$(echo "AF_PATH=${solibdir} /usr/local/fbcode/gcc-5-glibc-2.23/bin/mpirun -n ${nproc_per_node} ${training_script} --enable_distributed" \
      | /usr/local/chronos/scripts/crun --hostgroup "${hostgroup}" --gpu "${nproc_per_node}" --cpu "${cpu_per_node}" --secure-group oncall_fair_speech --timeout 29d)
else
  rand=$(echo -n $RANDOM | sha256sum | cut -c1-9)
  rndvdir="${rndvpath}/${rand}"
  # make a new directory with a random hash for rendezvous
  mkdir -m 777 "${rndvdir}"
  # copy the python script needed to determine world_rank into the rendezvous
  cp "${DIR}/launch_distributed_chronos_multinode.py" "${rndvdir}"
  jobid=$(echo "AF_PATH=${solibdir} /usr/local/fbcode/gcc-5-glibc-2.23/bin/mpirun -n ${nproc_per_node} python ${rndvdir}/launch_distributed_chronos_multinode.py ${training_script} --enable_distributed --world_size=${world_size} --rndv_filepath=${rndvdir}" \
      | /usr/local/chronos/scripts/crun --hostgroup "${hostgroup}" --gang-size "${nnodes}" --gang-rack-affinity --gpu "${nproc_per_node}" --cpu "${cpu_per_node}" --secure-group oncall_fair_speech --timeout 29d)
fi

echo "Job ID: ${jobid}"
echo "  https://our.intern.facebook.com/intern/bunny/?x+${jobid}"
