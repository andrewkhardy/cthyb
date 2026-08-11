/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2014, P. Seth, I. Krivenko, M. Ferrero and O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include "./util.hpp"
#include "./config.hpp" // many_body_op_t
#include <triqs/hilbert_space/hilbert_space.hpp>
#include <triqs/utility/time_pt.hpp>
#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/atom_diag/functions.hpp>
#include <triqs/utility/macros.hpp>

#include <h5/h5.hpp>

#include <map>

namespace triqs_cthyb {

  using triqs::utility::time_pt;
  using triqs::utility::time_segment;

  /// Description of a creation/annihilation operator.
  struct op_desc {
    /// Block index of the operator.
    int block_index;

    /// Inner index within the block.
    int inner_index;

    /// Whether the operator is a dagger (creation operator).
    bool dagger;

    /// Cumulative (linear) index.
    long linear_index;

    friend std::ostream &operator<<(std::ostream &out, op_desc const &op) {
      out << (op.dagger ? "Cdag(" : "C(") << op.block_index << "," << op.inner_index << ")";
      return out;
    }

    static std::string hdf5_format() { return "op_desc"; }

    friend void h5_write(h5::group g, std::string const &name, op_desc const &op) {
      auto gr = g.create_group(name);
      h5::write_hdf5_format(gr, op); // NOLINT (slicing is intended)
      h5::write(gr, "block", op.block_index);
      h5::write(gr, "inner", op.inner_index);
      h5::write(gr, "dagger", op.dagger);
      h5::write(gr, "linear_index", op.linear_index);
    }

    friend void h5_read(h5::group g, std::string const &name, op_desc &op) {
      h5::group gr = g.open_group(name);
      h5::assert_hdf5_format(gr, op);
      h5::read(g, "block", op.block_index);
      h5::read(g, "inner", op.inner_index);
      h5::read(g, "dagger", op.dagger);
      h5::read(g, "linear_index", op.linear_index);
    }

    bool operator==(op_desc const &op) const = default;
  };

  struct op_desc_pair_t { // NOLINT
    op_desc opL, opR;     // FIXME Need only block and inner index ? what about linear index ?

    bool operator==(op_desc_pair_t const &op) const = default;
    static std::string hdf5_format() { return "op_desc_pair_t"; }
    friend void h5_write(h5::group g, std::string const &name, op_desc_pair_t const &op) {
      auto gr = g.create_group(name);
      h5::write_hdf5_format(gr, op);
      h5::write(gr, "opL", op.opL);
      h5::write(gr, "opR", op.opR);
    }
    friend void h5_read(h5::group g, std::string const &name, op_desc_pair_t &op) {
      h5::group gr = g.open_group(name);
      h5::assert_hdf5_format(gr, op);
      h5::read(g, "opL", op.opL);
      h5::read(g, "opR", op.opR);
    }
  };

  struct bosonic_op_pair_t { // NOLINT
    op_desc_pair_t op1, op2;
    int f_index; // index of the function f associated to this pair

    bool operator==(bosonic_op_pair_t const &op) const = default;
    static std::string hdf5_format() { return "bosonic_op_pair_t"; }
    friend void h5_write(h5::group g, std::string const &name, bosonic_op_pair_t const &op) {
      auto gr = g.create_group(name);
      h5::write_hdf5_format(gr, op);
      h5::write(gr, "op1", op.op1);
      h5::write(gr, "op2", op.op2);
      h5::write(gr, "f_index", op.f_index);
    }
    friend void h5_read(h5::group g, std::string const &name, bosonic_op_pair_t &op) {
      h5::group gr = g.open_group(name);
      h5::assert_hdf5_format(gr, op);
      h5::read(g, "op1", op.op1);
      h5::read(g, "op2", op.op2);
      h5::read(g, "f_index", op.f_index);
    }
  };

  /// A single dynamical-interaction vertex, as specified by a user: a retarded coupling
  /// between two fermion bilinears, D(tau) * op1(tau) * op2(0).
  ///
  /// op1 and op2 are ordinary many-body operators (e.g. \c c_dag('up',0)*c('down',0)),
  /// exactly the same way \c h_int is specified -- NOT the low-level \c bosonic_op_pair_t
  /// above. Each is required to reduce to exactly one fermion bilinear c^dagger_a c_b; this
  /// is validated (via \c extract_bilinear, see dynamical_interactions.hpp) when the vertex
  /// is registered with the solver, not assumed. \c bosonic_op_pair_t is the internal
  /// representation the stochastic double expansion actually samples; a \c dyn_vertex_t is
  /// converted to it after the bilinear check.
  struct dyn_vertex_t { // NOLINT
    many_body_op_t op1, op2;
    gf<imtime, scalar_valued> coupling; // the retarded propagator D(tau) for this vertex

    static std::string hdf5_format() { return "dyn_vertex_t"; }
    friend void h5_write(h5::group g, std::string const &name, dyn_vertex_t const &v) {
      auto gr = g.create_group(name);
      h5::write_hdf5_format(gr, v);
      h5::write(gr, "op1", v.op1);
      h5::write(gr, "op2", v.op2);
      h5::write(gr, "coupling", v.coupling);
    }
    friend void h5_read(h5::group g, std::string const &name, dyn_vertex_t &v) {
      h5::group gr = g.open_group(name);
      h5::assert_hdf5_format(gr, v);
      h5::read(gr, "op1", v.op1);
      h5::read(gr, "op2", v.op2);
      h5::read(gr, "coupling", v.coupling);
    }
  };

  /// Configuration of the Monte Carlo simulation (operators on the imaginary-time line).
  struct configuration {

    bool operator==(configuration const &config) const { return (beta_ == config.beta_ && oplist_ == config.oplist_); }

    // a map associating an operator to an imaginary time
    using oplist_t = std::map<time_pt, op_desc, std::greater<time_pt>>;
    // @DYN_IMPL the couples of ops for J and \cal U
    // a list of pair or times + 2 monomial description (a,b,c,d) : c^+_a c_b   c_^+_c c_d
    //  f[f_index] (tau - tau') c^+_a c_b (tau) c^+_c d_d (tau')

    struct dyn_bosonic_pair_t { // NOLINT
      bosonic_op_pair_t ops;
      time_pt tau1, tau2;

      bool operator==(dyn_bosonic_pair_t const &op) const = default;
      static std::string hdf5_format() { return "dyn_bosonic_pair_t"; }
      friend void h5_write(h5::group g, std::string const &name, dyn_bosonic_pair_t const &op) {
        auto gr = g.create_group(name);
        h5::write_hdf5_format(gr, op);
        h5::write(gr, "ops", op.ops);
        h5::write(gr, "tau1", op.tau1);
        h5::write(gr, "tau2", op.tau2);
      }
      friend void h5_read(h5::group g, std::string const &name, dyn_bosonic_pair_t &op) {
        h5::group gr = g.open_group(name);
        h5::assert_hdf5_format(gr, op);
        h5::read(g, "ops", op.ops);
        h5::read(g, "tau1", op.tau1);
        h5::read(g, "tau2", op.tau2);
      }
    };
    using dyn_oplist_t = std::vector<dyn_bosonic_pair_t>;

#ifdef SAVE_CONFIGS
    configuration(double beta, long id = 0, oplist_t oplist = {})
       : beta_(beta), id_(id), oplist_(oplist), configs_hfile("configs.h5", exists("configs.h5") ? 'a' : 'w') {
      if (NUM_CONFIGS_TO_SAVE > 0) h5_write(configs_hfile, "c_0", *this);
    }
    ~configuration() { configs_hfile.close(); }
#else
    configuration(double beta, long id = 0) : beta_(beta), id_(id) {}
    C2PY_IGNORE configuration(double beta, long id, oplist_t oplist) : beta_(beta), id_(id), oplist_(oplist) {}
#endif

    /// Inverse temperature \f$ \beta \f$.
    C2PY_PROPERTY_GET(beta) double beta() const { return beta_; }
    auto size() const { return oplist_.size(); }

    /**
     * @brief Insert a given operator at a given imaginary time.
     * 
     * @param tau Imaginary time at which to insert the operator.
     * @param op Description of the operator to insert.
     */
    void insert(time_pt tau, op_desc op) { oplist_.insert({tau, op}); }

    /**
     * @brief Replace an existing operator at a given imaginary time with a new one.
     * 
     * @param tau Imaginary time at which to replace the operator.
     * @param op Description of the operator to insert.
     */
    void replace(time_pt tau, op_desc op) { oplist_[tau] = op; }

    /**
     * @brief Erase the operator at a given imaginary time.
     * @param tau Imaginary time at which to erase the operator.
     */
    void erase(time_pt const &t) { oplist_.erase(t); }

    /// Clear the configuration (remove all operators).
    void clear() { oplist_.clear(); }

    C2PY_IGNORE oplist_t::iterator find(time_pt const &t) { return oplist_.find(t); }
    C2PY_IGNORE oplist_t::const_iterator find(time_pt const &t) const { return oplist_.find(t); }

    C2PY_IGNORE oplist_t::iterator begin() { return oplist_.begin(); }
    C2PY_IGNORE oplist_t::iterator end() { return oplist_.end(); }
    C2PY_IGNORE oplist_t::const_iterator begin() const { return oplist_.begin(); }
    C2PY_IGNORE oplist_t::const_iterator end() const { return oplist_.end(); }

    // Find the n-th operator associated to an hybridiation in the configuration with given block_index and dagger
    C2PY_IGNORE time_pt find_nth_hybridization_op(int n, int block_index, bool dagger) {
      int i = 0;
      for (auto const &[tau, op] : oplist_)
        if (op.dagger == dagger && op.block_index == block_index && ++i == n + 1) return tau;
      TRIQS_RUNTIME_ERROR << "Operator not found";
    };

    friend std::ostream &operator<<(std::ostream &out, configuration const &c) {
      for (auto const &op : c) out << "tau = " << op.first << " : " << op.second << std::endl;
      return out;
    }

    /// HDF5 format string for configuration.
    static std::string hdf5_format() { return "CTHYB_Configuration"; }

    /// Write a configuration to an hdf5 file.
    friend void h5_write(h5::group g, std::string const &name, configuration const &c) {
      h5::group gr = g.create_group(name);
      h5::write_hdf5_format(gr, c); // NOLINT (slicing is intended)
      h5::write(gr, "beta", c.beta_);
      h5::write(gr, "id", c.id_);
      h5::write(gr, "oplist", c.oplist_);
      if (c.dyn_oplist.size() > 0) h5::write(gr, "dyn_oplist", c.dyn_oplist);
    }

    /// Read a configuration from an hdf5 file.
    C2PY_IGNORE static configuration h5_read_construct(h5::group g, std::string const &name) {
      h5::group gr = g.open_group(name);
      h5::assert_hdf5_format(gr, configuration{0.0});
      // h5::assert_hdf5_format<configuration>(gr);
      auto beta   = h5::read<double>(gr, "beta");
      auto id     = h5::read<long>(gr, "id");
      auto oplist = h5::read<oplist_t>(gr, "oplist");
      auto c = configuration(beta, id, std::move(oplist));
      if (gr.has_key("dyn_oplist")) h5::read(gr, "dyn_oplist", c.dyn_oplist);
      return c;
    }

    /// Get the ID of the current configuration (for debug purposes).
    C2PY_IGNORE long get_id() const { return id_; } // Get the id of the current configuration

    /// Finalize the configuration after a Monte Carlo move (increment the ID and save the configuration if needed).
    C2PY_IGNORE void finalize() {
      id_++;
#ifdef SAVE_CONFIGS
      if (id < NUM_CONFIGS_TO_SAVE) h5_write(configs_hfile, "c_" + std::to_string(id), *this);
#endif
    }

    //private:
    C2PY_IGNORE double beta_;
    C2PY_IGNORE long id_; // configuration id, for debug purposes
    C2PY_IGNORE oplist_t oplist_;
    C2PY_IGNORE dyn_oplist_t dyn_oplist;

#ifdef SAVE_CONFIGS
    // HDF5 file to save configurations
    h5::file configs_hfile;
#endif
  };
} // namespace triqs_cthyb
