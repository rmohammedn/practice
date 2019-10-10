#include<iostream>
#include<boost/function.hpp>
#include<boost/bind.hpp>

class Test;

typedef boost::function<int (Test*, int x, int y)> callback;
using callback2 = int(Test::*)(int, int);
int function(int x, int y, callback2 func, Test* t)
{
    return (t->*func)(x, y);
}

class Test
{
public:
    Test(){}
    int add(int x, int y)
    { return x + y;}
    int ros(int x, int y)
    {return x - y;}
};
int main()
{
    //boost::function<int (int x, int y)> f;
    callback f;
    callback2 f2;
    f2 = &Test::ros;
    Test *t;
    int x = 12, y = 21;
    f = &Test::add;
    std::cout << f(t, x, y) << std::endl;
    std::cout << function(x, y, f2, t) << std::endl;
    return 0;
}