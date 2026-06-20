#include <iostream>

#include "tiramisu/core/version.hpp"

int main() {
  std::cout << "tiramisu version: " << tiramisu::version_string() << "\n";
  return 0;
}