#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function

import os
import sys


"""
An intermediary script to parse environment variables set with OpenMPI
and Chronos' crun to determine a particular processes' world_rank
with respect to the node it's running on.

Executes all argv passed to the script with the addition of the
--world_rank flag as computed from gang id and mpi local size and rank.
"""


template = """
{training_command} \
--world_rank={world_rank}
"""


def run():
    # get OpenMPI environment variables
    local_size = os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE")
    local_rank = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
    gang_id = os.getenv("CHRONOS_GANG_MEMBER_ID")

    if local_size is None:
        raise Exception(
            "Could not parse MPI local size from "
            "environment variable OMPI_COMM_WORLD_LOCAL_SIZE"
        )

    if local_rank is None:
        raise Exception(
            "Could not parse MPI local rank from  "
            "environment bariable OMPI_COMM_WORLD_LOCAL_RANK"
        )

    if gang_id is None:
        raise Exception(
            "Could not parse gang ID from environment "
            "variable CHRONOS_GANG_MEMBER_ID"
        )

    world_rank = (int(local_size) * int(gang_id)) + int(local_rank)

    base_command_string = " ".join(sys.argv[1:])
    command = template.format(
        training_command=base_command_string, world_rank=world_rank
    )

    print(
        os.path.basename(__file__)
        + " on process of rank "
        + str(local_rank)
        + " running:\n",
        command,
    )
    # Run the training command, with all appended flags, as is
    os.system(command)


if __name__ == "__main__":
    run()
