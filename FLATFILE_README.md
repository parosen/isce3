# Flat Binary File Support for focus.py

This branch adds support for reading raw raster data from flat binary files while still using HDF5 for metadata.

## Overview

The modifications allow `focus.py` to:
- Read all metadata (chirp parameters, orbit, attitude, calibration, etc.) from HDF5
- Read raster data from external flat binary files
- Maintain full compatibility with existing HDF5-only workflows

## Use Cases

- Working with raw data stored in different formats
- Testing with synthetic data generated externally
- Interfacing with external processing systems
- Separating large raster data from metadata for storage optimization

## Changes Made

### 1. New Files

#### `python/packages/nisar/products/readers/Raw/BinaryDataDecoder.py`
- New class `BinaryDataDecoder` that mimics the `DataDecoder` interface
- Reads data from memory-mapped binary files
- Supports the same data types as HDF5: complex64, complex32, BFPQ
- Uses lookup tables from HDF5 for BFPQ decoding

#### `share/nisar/examples/focus_with_binary_files.yaml`
- Example configuration file showing how to specify binary file paths
- Documents the configuration structure

### 2. Modified Files

#### `python/packages/nisar/products/readers/Raw/Raw.py`
- Added `getRawDatasetFromBinary()` method
- Extracts metadata (shape, dtype, lookup tables) from HDF5
- Returns `BinaryDataDecoder` instance for binary file access

#### `python/packages/nisar/workflows/focus.py`
- Added logic to check for binary file configuration
- Falls back to HDF5 if no binary files specified
- Maintains backward compatibility

#### `python/packages/nisar/products/readers/Raw/__init__.py`
- Exports `BinaryDataDecoder` class

## Configuration

Add the following to your focus runconfig YAML:

```yaml
runconfig:
    groups:
        input_file_group:
            # HDF5 file for metadata (required)
            input_file_path:
            - /path/to/L0B_file.h5

            # Binary files for raster data (optional)
            binary_data_files:
                frequencyA:
                    HH: /path/to/raw_A_HH.bin
                    HV: /path/to/raw_A_HV.bin
                frequencyB:
                    VV: /path/to/raw_B_VV.bin
                    VH: /path/to/raw_B_VH.bin

            # Byte order (optional, default: 'native')
            binary_byte_order: native  # or 'little' or 'big'
```

## Binary File Format

Binary files must:
1. Match the shape specified in the HDF5 metadata
2. Use the same dtype as indicated in HDF5 (complex64, complex32, or BFPQ compound type)
3. Be in row-major (C) order
4. Have the correct byte order (specified in config)

### Example: Creating a Binary File

```python
import numpy as np
import h5py

# Read metadata from HDF5
with h5py.File('L0B_file.h5', 'r') as f:
    h5_data = f['/science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH']
    shape = h5_data.shape
    dtype = h5_data.dtype

# Create synthetic or processed data with same shape
data = np.zeros(shape, dtype=dtype)
# ... fill with actual data ...

# Write to binary file
data.tofile('raw_A_HH.bin')
```

## Backward Compatibility

- If `binary_data_files` is not specified in the config, the code uses HDF5 as before
- All existing runconfigs continue to work without modification
- No changes to output format or processing algorithms

## Testing

To test the implementation:

1. Extract raw data from an existing HDF5 file to binary format
2. Create a runconfig with `binary_data_files` pointing to the binary files
3. Run focus.py with the new config
4. Compare output with standard HDF5-based processing

## Limitations

- Binary files must exactly match the metadata in the HDF5 file
- No validation that binary file contents are correct
- Memory mapping requires sufficient virtual memory
- BFPQ lookup tables still come from HDF5

## Future Enhancements

Potential improvements:
- Support for metadata-only HDF5 files (without raster data)
- Validation of binary file dimensions
- Support for different binary layouts (column-major, tiled, etc.)
- Compression support for binary files
- Multiple binary files per polarization (chunked storage)
