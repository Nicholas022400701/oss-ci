# oss-ci

GitHub hosted test runner for my open source contribution branches.

Actions are disabled on forks by default and some upstream projects only run
their suites on self hosted runners, so this repository provides one
`workflow_dispatch` workflow that checks out any repository at any ref and runs
the install and test commands given at dispatch time. Each run is named after
the repository and ref it tested.
