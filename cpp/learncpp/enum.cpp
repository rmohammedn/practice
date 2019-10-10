# include <iostream>

enum color
{
    WHITE,
    RED,
    BLUE = 3,
    BLACK,
};

int main()
{
    color sky(RED);
    std::cout << sky << std::endl;
    return 0;
}