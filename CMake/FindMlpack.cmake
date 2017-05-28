# Store the mlpack include directory in MLPACK_INCLUDE_DIR.
# The PATHS variable may be specified to give hints for where to find core.hpp.
find_path(MLPACK_INCLUDE_DIR
  NAMES core.hpp prereqs.hpp
  PATHS "$ENV{ProgramFiles}/mlpack/" /usr/local/include/
)

# Find libmlpack.so (or equivalent) and store it in MLPACK_LIBRARY.
# If this example script were smarter, it would also find other dependencies of mlpack and store them in
# an MLPACK_LIBRARIES variable instead.
find_library(MLPACK_LIBRARY
  NAMES mlpack
  PATHS "$ENV{ProgramFiles}/mlpack/" /usr/lib64/ /usr/lib/ /usr/local/lib64/ /usr/local/
)