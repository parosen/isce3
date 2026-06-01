import logging
import numpy as np
from isce3.io import decode_bfpq_lut
from isce3.core.types import complex32, read_c4_dataset_as_c8

log = logging.getLogger("Raw")

class BinaryDataDecoder(object):
    """Handle reading raw data from flat binary files instead of HDF5.
    
    This class mimics the interface of DataDecoder but reads from a memory-mapped
    binary file. Metadata (shape, dtype, lookup tables) still comes from HDF5.
    Indexing operations always return data converted to complex64.
    
    Parameters
    ----------
    binary_path : str
        Path to the flat binary file containing raw data
    shape : tuple
        Shape of the data array (azimuth, range)
    dtype_storage : numpy.dtype
        Storage dtype in the binary file (e.g., complex32, complex64, or BFPQ compound type)
    lut_table : numpy.ndarray, optional
        BFPQ lookup table for decoding, if applicable
    byte_order : str, optional
        Byte order of binary file ('native', 'little', 'big'). Default is 'native'.
    """
    
    def __init__(self, binary_path, shape, dtype_storage, lut_table=None, byte_order='native'):
        self.binary_path = binary_path
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = np.dtype('c8')  # Output is always complex64
        self.dtype_storage = dtype_storage
        self.table = lut_table
        self.read_count = 0  # Track number of reads from binary file
        
        log.info("="*80)
        log.info("BINARY FILE MODE: Using BinaryDataDecoder instead of HDF5 DataDecoder")
        log.info(f"Binary file path: {binary_path}")
        log.info(f"Data shape from HDF5 metadata: {shape}")
        log.info(f"Storage dtype: {dtype_storage}")
        log.info(f"Byte order: {byte_order}")
        log.info("="*80)
        
        # Adjust dtype for byte order if needed
        if byte_order == 'little':
            dtype_for_mmap = dtype_storage.newbyteorder('<')
        elif byte_order == 'big':
            dtype_for_mmap = dtype_storage.newbyteorder('>')
        else:
            dtype_for_mmap = dtype_storage
        
        # Memory-map the binary file for efficient access
        try:
            self.data_mmap = np.memmap(binary_path, dtype=dtype_for_mmap, 
                                       mode='r', shape=shape)
            log.info(f"Memory-mapped binary file {binary_path} with shape {shape}")
        except Exception as e:
            raise IOError(f"Failed to memory-map binary file {binary_path}: {e}")
        
        # Set up the appropriate decoder based on data type
        if self.table is not None:
            self.decoder = self._decode_lut
            log.info("Will decode binary data with BFPQ lookup table.")
        elif self.dtype_storage == complex32:
            self.decoder = self._decode_complex32
            log.info("Will decode binary data from float16 encoding.")
        elif self.dtype_storage == np.complex64:
            self.decoder = lambda key: self.data_mmap[key]
            log.info("Binary data is complex64, no decoding required.")
        else:
            raise ValueError(f"Unsupported binary data type {self.dtype_storage}")
    
    def __getitem__(self, key):
        """Return decoded data for the given slice/index."""
        self.read_count += 1
        
        # Log first few reads and periodic updates to show binary file is being used
        if self.read_count <= 5 or self.read_count % 100 == 0:
            log.info(f"BinaryDataDecoder read #{self.read_count}: Reading slice {key} from binary file")
            if self.read_count == 1:
                log.info(f"  -> First read from binary file: {self.binary_path}")
        
        result = self.decoder(key)
        
        # Log details of first read
        if self.read_count == 1:
            log.info(f"  -> Read result shape: {result.shape}, dtype: {result.dtype}")
            log.info(f"  -> Sample values: min={np.min(np.abs(result)):.6e}, max={np.max(np.abs(result)):.6e}, mean={np.mean(np.abs(result)):.6e}")
        
        return result
    
    def _decode_lut(self, key):
        """Decode BFPQ data using lookup table."""
        z = self.data_mmap[key]
        assert self.table is not None
        # Only have 2D version in C++, fall back to Python otherwise
        if z.ndim == 2:
            return decode_bfpq_lut(self.table, z)
        else:
            return self.table[z['r']] + 1j * self.table[z['i']]
    
    def _decode_complex32(self, key):
        """Decode complex32 (float16 pairs) to complex64."""
        z = self.data_mmap[key]
        # Convert from complex32 (pairs of float16) to complex64
        if hasattr(z, 'dtype') and z.dtype == complex32:
            # Use the same approach as read_c4_dataset_as_c8
            # but for numpy arrays instead of h5py datasets
            return z.astype(np.complex64)
        return z
    
    def __del__(self):
        """Clean up memory-mapped file."""
        if hasattr(self, 'data_mmap'):
            del self.data_mmap
