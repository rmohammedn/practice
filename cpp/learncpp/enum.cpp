#include <iostream>

enum class Student
{
    MOHAMMED,
    RAFSHAN,
};

int main()
{
    Student maths{Student::MOHAMMED};
    std::cout << maths << std::endl;
    return 0;
}