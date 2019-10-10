#include <math.h>

class Math
{
    public:
    double squareRoot(const double x);
};

double Math::squareRoot(const double x)
{
    if (x < 0)
        return -1;
    else
        return sqrt(x);

}