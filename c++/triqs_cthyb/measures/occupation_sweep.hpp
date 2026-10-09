#pragma once

#include <utility>
#include <vector>
#include <nda/nda.hpp>
#include <mpi/mpi.hpp>
#include <triqs/mesh.hpp>

#include "../qmc_data.hpp"

namespace triqs_cthyb {

  /// <A_i(tau) B_j(0)> for bosonic operators (an even number of c and c^dagger in every term), evaluated exactly at the
  /// nodes of a DLR imaginary-time grid, configuration by configuration.
  ///
  /// B_j sits at tau = 0 and A_i at each node. Between the trace operators at t_k and t_{k+1},
  ///   <A_i(tau) B_j(0)> = sum_mn (A_i)_mn e^{-(t_{k+1} - tau) E_m} e^{-(tau - t_k) E_n} (R_k B_j L_k)_nm / trace,
  /// with R_k the trace from 0 to t_k and L_k the trace from t_{k+1} to beta. Only the intervals that hold a node are
  /// evaluated.
  ///
  /// Operators that map every block of h_diag to itself (n_a, S_z, ...) follow the blocks of the configuration. The
  /// others (S^+, c^dagger_a c_b, ...) need R_k and L_k from every block, and when they change densities that a
  /// Lang-Firsov vertex couples, the Lang-Firsov weight exp(phi_A(tau) + phi_B(0) + A-B interaction) of their kinks.
  class occupation_sweep {
    public:
    /// nodes must be symmetrized (the default of dlr_imtime), i.e. mapped onto themselves by tau -> beta - tau
    occupation_sweep(qmc_data const &data, triqs::mesh::dlr_imtime const &nodes, std::vector<many_body_op_t> const &ops_tau,
                     std::vector<many_body_op_t> const &ops_0);

    void accumulate(mc_weight_t s);

    /// (node, i, j) -> <A_i(tau_node) B_j(0)>, nodes in the order of the mesh, all-reduced over c
    nda::array<mc_weight_t, 3> collect(mpi::communicator const &c);

    /// The node at beta - tau of node l, both in the order of the mesh
    long reflected(long l) const { return reflection[l]; }

    private:
    // A part of operator op that maps every block of h_diag to at most one block, with the kinks kink_vectors[kinks] in
    // all its terms
    struct part_t {
      long op;
      atom_diag::op_block_mat_t mat;
      int kinks;
    };
    struct term_t {
      int m, n; // a matrix element of a block of h_diag where some A_i is nonzero
    };

    // Add the measurement to values, and return the bare trace and the sum of the absolute values of its blocks
    std::pair<h_scalar_t, double> accumulate_within_blocks(mc_weight_t weight);
    std::pair<h_scalar_t, double> accumulate_across_blocks(mc_weight_t weight);

    // The matrix of the trace operator k from block b, and the block it maps b to
    matrix_t const &op_matrix(int k, int b) const;
    long op_target(int k, int b) const;

    qmc_data const &data;
    long n_A, n_B;
    bool within_blocks;                                  // every A_i and B_j maps each block of h_diag to itself
    std::vector<atom_diag::op_block_mat_t> A_mat, B_mat; // within_blocks: in the eigenbasis, by block of h_diag
    std::vector<std::vector<term_t>> terms;              // within_blocks: by block of h_diag
    std::vector<part_t> A_parts, B_parts;                // otherwise
    std::vector<qmc_data::kinks_t> kink_vectors;         // of the parts; [0] is empty, for kinks the Lang-Firsov kernel does not see
    std::vector<double> node_tau;                        // the nodes in increasing tau
    std::vector<long> mesh_index;                        // the index in the mesh of each of them
    std::vector<long> reflection;                        // by index in the mesh
    nda::array<mc_weight_t, 2> values;                   // (node in increasing tau, i * n_B + j)
    mc_weight_t average_sign = 0;

    // Work space, kept between measurements
    timed_ops_t ops; // the trace operators in increasing time
    std::vector<double> t;
    std::vector<long> first_node;
    std::vector<int> block;
    std::vector<std::vector<double>> decays;
    std::vector<matrix_t> R, L, Y;
    matrix_t evolved, RB, Y_part;
    nda::matrix<double> left, right; // (node of the interval, state): e^{-(t_{k+1} - tau) E}, e^{-(tau - t_k) E}
    std::vector<mc_weight_t> coefficients;
    std::vector<int> slot_of;                      // across blocks: the slot of each interval that holds a node, else -1
    std::vector<std::vector<matrix_t>> R_at, L_at; // across blocks: (slot, block the product starts in)
    std::vector<std::vector<long>> R_block, L_end; // the block it reaches, -1 if the trace vanishes on the way
    std::vector<matrix_t> L_level, L_next;
    std::vector<long> end_level, end_next;
    nda::matrix<double> phi_A; // (node in increasing tau, kinks): the Lang-Firsov potential of the kinks of A at the node
    std::vector<double> phi_B; // (kinks): and of those of B at 0
  };

} // namespace triqs_cthyb
