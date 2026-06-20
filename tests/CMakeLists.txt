add_executable(tiramisu_tests
  core/version_test.cpp
)

target_link_libraries(tiramisu_tests PRIVATE
  tiramisu_core
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tiramisu_tests)