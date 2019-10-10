#include<iostream>
#include<gtest/gtest.h>
#include "test.h"

int function(int x, int y)
{
    return x*y;
}

class whoami : public ::testing::Test
{
public:
    int add(int x, int y);
    int sub(int x, int y); 
};

class Check : public ::testing::Test
{
protected:
    int add(int x, int y);
    int sub(int x, int y);
    Math m;
};

TEST(testNumber1, testType1)
{
    Math m;
    EXPECT_EQ(4, m.squareRoot(16));
    EXPECT_NE(4, m.squareRoot(25));
    EXPECT_EQ(-1, m.squareRoot(-2));
}
TEST(whoami, testType2)
{
    whoami i;
    EXPECT_EQ(4, i.add(2, 2));
    EXPECT_NE(5, function(2, 2));
    EXPECT_GT(5, function(2, 1));
}
TEST_F(Check, testType1)
{
    EXPECT_EQ(4, m.squareRoot(16));
    EXPECT_NE(5, m.squareRoot(15));
    EXPECT_GT(5, 3);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}