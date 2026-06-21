#include <gtest/gtest.h>
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/ops/broadcast.hpp"
#include "tiramisu/ops/elementwise.hpp"
#include "tiramisu/ops/matmul.hpp"
#include "tiramisu/ops/reduce.hpp"

TEST(OpsTest, BroadcastShapes) {
    EXPECT_EQ(tiramisu::ops::broadcast_shapes({3, 4}, {3, 4}), std::vector<int64_t>({3, 4}));
    EXPECT_EQ(tiramisu::ops::broadcast_shapes({3, 4}, {1}), std::vector<int64_t>({3, 4}));
    EXPECT_EQ(tiramisu::ops::broadcast_shapes({3, 4}, {1, 4}), std::vector<int64_t>({3, 4}));
    EXPECT_EQ(tiramisu::ops::broadcast_shapes({3, 4}, {3, 1}), std::vector<int64_t>({3, 4}));
    EXPECT_EQ(tiramisu::ops::broadcast_shapes({2, 3, 4}, {4}), std::vector<int64_t>({2, 3, 4}));
    EXPECT_THROW(tiramisu::ops::broadcast_shapes({3, 4}, {3, 5}), std::invalid_argument);
}


TEST(OpsTest, AddSameShape) {
    tiramisu::Tensor a({2, 2});
    tiramisu::Tensor b({2, 2});

    a.at<float>({0, 0}) = 1; a.at<float>({0, 1}) = 2;
    a.at<float>({1, 0}) = 3; a.at<float>({1, 1}) = 4;

    b.at<float>({0, 0}) = 5; b.at<float>({0, 1}) = 6;
    b.at<float>({1, 0}) = 7; b.at<float>({1, 1}) = 8;

    tiramisu::Tensor c = tiramisu::ops::add(a, b);

    EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 6);
    EXPECT_FLOAT_EQ(c.at<float>({0, 1}), 8);
    EXPECT_FLOAT_EQ(c.at<float>({1, 0}), 10);
    EXPECT_FLOAT_EQ(c.at<float>({1, 1}), 12);
}

TEST(OpsTest, AddBroadcast) {
    tiramisu::Tensor matrix({2, 3});
    tiramisu::Tensor row_vector({1, 3});

    // Matrix:
    // 1 2 3
    // 4 5 6
    matrix.at<float>({0, 0}) = 1; matrix.at<float>({0, 1}) = 2; matrix.at<float>({0, 2}) = 3;
    matrix.at<float>({1, 0}) = 4; matrix.at<float>({1, 1}) = 5; matrix.at<float>({1, 2}) = 6;

    // Vector: 10 20 30
    row_vector.at<float>({0, 0}) = 10;
    row_vector.at<float>({0, 1}) = 20;
    row_vector.at<float>({0, 2}) = 30;

    // Should broadcast row_vector to both rows
    tiramisu::Tensor c = tiramisu::ops::add(matrix, row_vector);

    EXPECT_EQ(c.shape(), std::vector<int64_t>({2, 3}));
    EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 11);
    EXPECT_FLOAT_EQ(c.at<float>({1, 2}), 36);
}

TEST(OpsTest, SumAndMean) {
    tiramisu::Tensor a({4});
    a.at<float>({0}) = 2;
    a.at<float>({1}) = 4;
    a.at<float>({2}) = 6;
    a.at<float>({3}) = 8;

    tiramisu::Tensor s = tiramisu::ops::sum(a);
    tiramisu::Tensor m = tiramisu::ops::mean(a);

    EXPECT_FLOAT_EQ(s.at<float>({0}), 20);
    EXPECT_FLOAT_EQ(m.at<float>({0}), 5);
}

TEST(OpsTest, Matmul) {
    // 2x3
    tiramisu::Tensor a({2, 3});
    a.at<float>({0, 0}) = 1; a.at<float>({0, 1}) = 2; a.at<float>({0, 2}) = 3;
    a.at<float>({1, 0}) = 4; a.at<float>({1, 1}) = 5; a.at<float>({1, 2}) = 6;

    // 3x2
    tiramisu::Tensor b({3, 2});
    b.at<float>({0, 0}) = 7;  b.at<float>({0, 1}) = 8;
    b.at<float>({1, 0}) = 9;  b.at<float>({1, 1}) = 10;
    b.at<float>({2, 0}) = 11; b.at<float>({2, 1}) = 12;

    tiramisu::Tensor c = tiramisu::ops::matmul(a, b);

    EXPECT_EQ(c.shape(), std::vector<int64_t>({2, 2}));
    
    // (1*7 + 2*9 + 3*11) = 58
    EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 58);
    // (1*8 + 2*10 + 3*12) = 64
    EXPECT_FLOAT_EQ(c.at<float>({0, 1}), 64);
    // (4*7 + 5*9 + 6*11) = 139
    EXPECT_FLOAT_EQ(c.at<float>({1, 0}), 139);
    // (4*8 + 5*10 + 6*12) = 154
    EXPECT_FLOAT_EQ(c.at<float>({1, 1}), 154);
}

TEST(OpsTest, BinaryMath) {
    tiramisu::Tensor a({2}); a.at<float>({0}) = 10; a.at<float>({1}) = 20;
    tiramisu::Tensor b({2}); b.at<float>({0}) = 2;  b.at<float>({1}) = 5;

    auto t_sub = tiramisu::ops::sub(a, b);
    EXPECT_FLOAT_EQ(t_sub.at<float>({0}), 8);
    EXPECT_FLOAT_EQ(t_sub.at<float>({1}), 15);

    auto t_div = tiramisu::ops::div(a, b);
    EXPECT_FLOAT_EQ(t_div.at<float>({0}), 5);
    EXPECT_FLOAT_EQ(t_div.at<float>({1}), 4);
}

TEST(OpsTest, UnaryMath) {
    tiramisu::Tensor a({3});
    a.at<float>({0}) = -5.0f;
    a.at<float>({1}) = 0.0f;
    a.at<float>({2}) = 3.0f;

    auto t_neg = tiramisu::ops::neg(a);
    EXPECT_FLOAT_EQ(t_neg.at<float>({0}), 5.0f);
    EXPECT_FLOAT_EQ(t_neg.at<float>({2}), -3.0f);

    auto t_relu = tiramisu::ops::relu(a);
    EXPECT_FLOAT_EQ(t_relu.at<float>({0}), 0.0f);
    EXPECT_FLOAT_EQ(t_relu.at<float>({2}), 3.0f);

    // Exp / Log test
    tiramisu::Tensor b({1}); b.at<float>({0}) = 1.0f;
    auto t_exp = tiramisu::ops::exp(b);
    EXPECT_NEAR(t_exp.at<float>({0}), 2.71828f, 1e-4);
    
    auto t_log = tiramisu::ops::log(t_exp);
    EXPECT_NEAR(t_log.at<float>({0}), 1.0f, 1e-4);
}