"""

Minimal replacement for the removed Python standard library ``chunk`` module.

The :mod:`chunk` module was deprecated in Python 3.11 and removed in Python
3.13.  PyCozmo only needs a small subset of its functionality to walk RIFF
chunks in AudioKinetic Wwise sound banks, so this module provides a compatible,
self-contained implementation.

"""

import struct


__all__ = [
    "Chunk",
]


class Chunk:
    """A RIFF chunk.

    Args:
        file: A seekable binary file object.
        align: Whether to align the end of the chunk to an even byte boundary.
        bigendian: Whether the chunk length is big-endian.
    """

    def __init__(self, file, align=True, bigendian=False):
        self.file = file
        self.align = align
        self.bigendian = bigendian
        self.offset = file.tell()

        header = file.read(8)
        if len(header) < 8:
            raise EOFError("Not enough data for chunk header")

        self.chunkname = header[:4]
        self.chunksize = struct.unpack(">L" if bigendian else "<L", header[4:])[0]
        self.size_read = 0
        self.chunkstart = self.offset + 8
        self.closed = False

    def getname(self):
        """Return the chunk identifier as bytes."""
        return self.chunkname

    def getsize(self):
        """Return the chunk size in bytes."""
        return self.chunksize

    def read(self, size=-1):
        """Read up to ``size`` bytes from the chunk."""
        if self.closed:
            raise ValueError("I/O operation on closed chunk")
        if size is None or size < 0:
            size = self.chunksize - self.size_read
        else:
            size = min(size, self.chunksize - self.size_read)
        data = self.file.read(size)
        self.size_read += len(data)
        return data

    def skip(self):
        """Skip the rest of the chunk and prepare the file for the next chunk."""
        if self.closed:
            return
        remaining = self.chunksize - self.size_read
        if remaining > 0:
            try:
                self.file.seek(remaining, 1)
            except (OSError, ValueError):
                self.file.read(remaining)
        if self.align and self.chunksize & 1:
            try:
                self.file.seek(1, 1)
            except (OSError, ValueError):
                self.file.read(1)
        self.size_read = self.chunksize
        self.file.seek(self.chunkstart + self.chunksize, 0)
        self.closed = True

    def close(self):
        """Close the chunk, skipping any unread data."""
        self.skip()

    def tell(self):
        """Return the current position inside the chunk."""
        return self.size_read

    def seek(self, offset, whence=0):
        """Seek inside the chunk."""
        if self.closed:
            raise ValueError("I/O operation on closed chunk")
        if whence == 0:
            new_pos = offset
        elif whence == 1:
            new_pos = self.size_read + offset
        elif whence == 2:
            new_pos = self.chunksize + offset
        else:
            raise ValueError("Invalid whence value")
        if new_pos < 0:
            raise OSError("Invalid argument")
        new_pos = min(new_pos, self.chunksize)
        self.file.seek(self.chunkstart + new_pos, 0)
        self.size_read = new_pos
        return new_pos

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
