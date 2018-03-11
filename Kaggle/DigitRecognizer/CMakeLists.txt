# Recurse into each model.
set(DIRS
  # Model directory.
  src/FNN/
  src/CNN/
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()
