// Spawn N threads. Each thread prints "Hello from thread X of N" where X is the thread’s
// ID (0-based). Join all threads
# include <iostream>
# include <thread>
# include <vector>
void hello ( int id , int total ) {
    std::cout << "Hello from thread " << id << " of " << total << " \n" ;
}

int main () {
    const int N = 5 ;
    std::vector < std::thread > threads ;
    threads.reserve(N) ;
    for (int i = 0 ; i < N ; ++ i )
        threads.emplace_back (hello, i, N) ;
    for (auto &t : threads ) t.join () ;
    return 0;
}
