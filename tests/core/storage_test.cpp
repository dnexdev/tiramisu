#include <gtest/gtest.h>

#include <cstdint>
#include <cstddef>

#include "tiramisu/core/storage.hpp"
#include "tiramisu/core/device.hpp"
#include "tiramisu/core/dtype.hpp"

TEST(StorageTest, BasicProperties) {
  tiramisu::DType dtype = tiramisu::DType::Float32;
  tiramisu::Device device = tiramisu::Device::CPU;
  std::size_t count = 100;
  
  tiramisu::Storage storage(count, dtype, device);

  EXPECT_EQ(storage.numel(), count);
  EXPECT_EQ(storage.dtype(), dtype);
  EXPECT_EQ(storage.device(), device);
  EXPECT_NE(storage.data(), nullptr);
  
  // verify nbytes matches the count * size of the dtype
  EXPECT_EQ(storage.nbytes(), count * tiramisu::dtype_size(dtype));
}

TEST(StorageTest, EmptyStorageHandlesNullptr) {
  tiramisu::DType dtype = tiramisu::DType::Float32;
  tiramisu::Device device = tiramisu::Device::CPU;
  
  tiramisu::Storage storage(0, dtype, device);

  EXPECT_EQ(storage.numel(), 0);
  EXPECT_EQ(storage.nbytes(), 0);
  
  // 0-count allocations should immediately return nullptr to save overhead
  EXPECT_EQ(storage.data(), nullptr);
}

TEST(StorageTest, RespectsAlignment) {
  tiramisu::DType dtype = tiramisu::DType::Float32;
  tiramisu::Device device = tiramisu::Device::CPU;
  
  // Test default alignment (64 bytes)
  tiramisu::Storage storage_default(10, dtype, device);
  auto ptr_val_default = reinterpret_cast<std::uintptr_t>(storage_default.data());
  EXPECT_EQ(ptr_val_default % 64, 0) << "Default 64-byte alignment failed.";

  // Test custom strict alignment (e.g., 256 bytes for AVX/specific hardware constraints)
  tiramisu::Storage storage_custom(10, dtype, device, 256);
  auto ptr_val_custom = reinterpret_cast<std::uintptr_t>(storage_custom.data());
  EXPECT_EQ(ptr_val_custom % 256, 0) << "Custom 256-byte alignment failed.";
}

TEST(StorageTest, ReadWriteMemorySafety) {
  tiramisu::DType dtype = tiramisu::DType::Float32;
  tiramisu::Device device = tiramisu::Device::CPU;

  tiramisu::Storage storage(1024, dtype, device);
  std::byte* data = storage.data();
  
  // Use ASSERT_NE before writing to avoid segfaults in the test runner
  ASSERT_NE(data, nullptr);

  for (std::size_t i = 0; i < storage.nbytes(); ++i) {
    data[i] = static_cast<std::byte>(i % 256);
  }

  const tiramisu::Storage& const_storage = storage;
  const std::byte* cdata = const_storage.data();
  
  for (std::size_t i = 0; i < storage.nbytes(); ++i) {
    EXPECT_EQ(cdata[i], static_cast<std::byte>(i % 256));
  }
}