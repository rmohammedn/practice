#include <thread>
#include <iostream>
#include <mutex>

void threadFunction()
{
    std::mutex m;
    while(1)
    {
        m.unlock();
        std::cout << "I am here" << std::endl;
        m.lock();
    }
}

void threadFunc()
{
    std::mutex m;
    while(1)
    {
        m.unlock();
        std::cout << "I am where" << std::endl;
        m.lock();
    }
}

int main()
{
    std::thread th(threadFunction);
    std::thread th2(threadFunc);
    // th.detach();
    // if (th.joinable())
    // {
    //     th.join();
    // }
    th.join();
    th2.join();
    return 0;
}