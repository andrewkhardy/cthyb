#include <triqs/gfs.hpp>
using namespace triqs::gfs;
int main() {
    auto b = make_block2_gf<legendre>({10.0, Boson, 100}, gf_struct_t{{"up", 1}, {"down", 1}});
    std::cout << "Success!" << std::endl;
    return 0;
}
