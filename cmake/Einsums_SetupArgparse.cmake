#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

# Set options for argparse before fetching
set(ARGPARSE_INSTALL ON CACHE BOOL "Enable argparse installation" FORCE)

fetchcontent_declare(
  argparse
  GIT_REPOSITORY https://github.com/Einsums/argparse.git
  GIT_TAG master
  OVERRIDE_FIND_PACKAGE
)
fetchcontent_makeavailable(argparse)
