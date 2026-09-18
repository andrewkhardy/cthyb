# Configure the version
configure_file(version.py.in version.py)

# All Python files. Copy them in the build dir to have a complete package for the tests.
file(GLOB_RECURSE python_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
foreach(file ${python_sources})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# Install python files to proper location
set(PYTHON_LIB_DEST ${TRIQS_PYTHON_LIB_DEST_ROOT}/${PROJECT_NAME})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.py DESTINATION ${PYTHON_LIB_DEST})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT} FILES_MATCHING PATTERN "*.py")

# The Python bindings (*.wrap.cxx / *.wrap.hxx) are GENERATED from the C++ headers by
# clair-c2py. Never edit them by hand: a hand-patched wrapper silently drifts from
# solver_core.hpp and then fails to compile against it (e.g. it kept binding
# lang_firsov_U_renorm / lang_firsov_mu_renorm after those members were removed).
# So regenerate by default wherever the generator exists, which makes the headers the
# single source of truth and the post-build git diff the record of what changed.
# Where clair-c2py is absent (CI, downstream builds) fall back to the committed wrappers.
find_program(CLAIR_C2PY_EXECUTABLE clair-c2py)
if(CLAIR_C2PY_EXECUTABLE)
  # FORCE, so a stale cache from an earlier OFF configure cannot silently reintroduce drift.
  set(Update_Python_Bindings ON CACHE BOOL "Regenerate the C++ Python bindings via clair-c2py" FORCE)
else()
  set(Update_Python_Bindings OFF CACHE BOOL "Regenerate the C++ Python bindings via clair-c2py" FORCE)
  message(STATUS "clair-c2py not found: building against the committed *.wrap.cxx. "
                 "Do not edit those by hand -- install clair-c2py to regenerate them.")
endif()

# Fetch c2py (provides c2py_add_module)
include(FetchContent)
FetchContent_Declare(
  c2py
  GIT_REPOSITORY  https://github.com/flatironinstitute/c2py
  GIT_TAG         unstable
  EXCLUDE_FROM_ALL # exclude from all so that c2py is NOT installed.
)
FetchContent_MakeAvailable(c2py)

# Add Python modules
c2py_add_module(configuration LINK_LIBRARIES ${PROJECT_NAME}_c ${PROJECT_NAME}_warnings)
c2py_add_module(solver_core LINK_LIBRARIES ${PROJECT_NAME}_c ${PROJECT_NAME}_warnings DEPENDS_ON_BINDINGS configuration)
install(TARGETS configuration solver_core DESTINATION ${PYTHON_LIB_DEST})
