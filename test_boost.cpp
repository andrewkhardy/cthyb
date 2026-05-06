#include <iostream>
#include <boost/math/special_functions/legendre.hpp>
int main() { std::cout << boost::math::legendre_p(2, 0.5) << std::endl; }
