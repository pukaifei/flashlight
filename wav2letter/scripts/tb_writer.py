# Modified from https://fburl.com/mj3xplvt

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import time
import struct
from crc32c import crc32


class Writer:

    def __init__(self, outdir, filename=None):
        # tensorboard looks for tag "tfevents" in filename to load data
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if filename is None:
            filename = "events.out.tfevents." + str(int(time.time()))
        self.writer = open(os.path.join(outdir, filename), 'wb')
        # every log file has to start with event of file version
        self.writeEvent(
            tf.Event(
                wall_time=time.time(),
                step=0,
                file_version='brain.Event:2'))

    # this function replicates scalar() function in tensorflow, simlpy logs a
    # single value and plot in a graph
    def write(self, summary, step, time):
        # data will wrap in summary and write as a Event protobuf
        #'tag' will group the plot data in a single graph
        event = tf.Event(
            wall_time=time,
            step=step,
            summary=summary)

        self.writeEvent(event)

    def writeEvent(self, event):
        # serialize the protobuf as a string
        data = event.SerializeToString()
        w = self.writer
        # tensorboard uses a checksum algorithm(CRC) to verify data integrity

        #format defined in here: https://fburl.com/4jwq6z4e

        # Format of a single record:
        # uint64    length
        # uint32    masked crc of length
        # byte      data[length]
        # uint32    masked crc of data

        # struck.pack will format string as binary data in a format
        # 'Q' is the format of unsigned long long(uint64)
        header = struct.pack('Q', len(data))
        w.write(header)
        # 'I' is unsigned int(uint32)
        w.write(struct.pack('I', masked_crc32c(header)))
        w.write(data)
        w.write(struct.pack('I', masked_crc32c(data)))
        w.flush()

    def close(self):
        self.writer.close()


def masked_crc32c(data):
    # mast function defined in: https://fburl.com/46l2ffqn
    kMaskDelta = 0xa282ead8
    x = u32(crc32(data))
    return u32(((x >> 15) | u32(x << 17)) + kMaskDelta)


def u32(x):
    return x & 0xffffffff
