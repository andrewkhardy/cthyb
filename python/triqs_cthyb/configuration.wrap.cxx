
// C.f. https://numpy.org/doc/1.21/reference/c-api/array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL _cpp2py_ARRAY_API
#ifndef CLAIR_C2PY_WRAP_GEN
#ifdef __clang__
// #pragma clang diagnostic ignored "-W#warnings"
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wcpp"
#endif

#define C2PY_VERSION_MAJOR 1
#define C2PY_VERSION_MINOR 0

#include <c2py/c2py.hpp>
#include <c2py/version_check.hpp>
#include <c2py/serialization/h5.hpp>

using c2py::operator""_a;

// ==================== enums =====================

// ==================== module classes =====================

// --------- class _c2py_cls_6805ae04 -----------
using _c2py_cls_6805ae04                                            = triqs_cthyb::op_desc;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_6805ae04>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_6805ae04> = "triqs_cthyb.configuration.OpDesc";

static int synth_constructor_7e5d6987(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (args and PyTuple_Check(args) and (PyTuple_Size(args) > 0)) {
    PyErr_SetString(PyExc_RuntimeError, ("Error in constructing triqs_cthyb::op_desc.\nNo positional arguments allowed. Use keywords arguments"));
    return -1;
  }
  c2py::pydict_extractor de{kwargs};
  try {
    ((c2py::wrap<_c2py_cls_6805ae04> *)self)->_c = new _c2py_cls_6805ae04{};
  } catch (std::exception const &e) {
    PyErr_SetString(PyExc_RuntimeError, ("Error in constructing triqs_cthyb::op_desc from a Python dict.\n   "s + e.what()).c_str());
    return -1;
  }
  auto &self_c = *(((c2py::wrap<_c2py_cls_6805ae04> *)self)->_c);
  de("block_index", self_c.block_index, false);
  de("inner_index", self_c.inner_index, false);
  de("dagger", self_c.dagger, false);
  de("linear_index", self_c.linear_index, false);
  return de.check();
}

template <> constexpr initproc c2py::tp_init<_c2py_cls_6805ae04> = synth_constructor_7e5d6987;

template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_6805ae04> =
   c2py::replace_tags(R"DOC(Synthesized constructor with the following keyword arguments:

Parameters
----------
block_index : {par_0}

inner_index : {par_1}

dagger : {par_2}

linear_index : {par_3}

)DOC",
                      "par",
                      {c2py::python_typename<int>(), c2py::python_typename<int>(), c2py::python_typename<bool>(), c2py::python_typename<long>()});

// ----- Method table ----
// clang-format off
template <> PyMethodDef c2py::tp_methods<_c2py_cls_6805ae04>[] = {
   {"__write_hdf5__", c2py::tpxx_write_h5<_c2py_cls_6805ae04>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_h5<_c2py_cls_6805ae04>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_h5<_c2py_cls_6805ae04>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};
// clang-format on

constexpr auto _c2py_doc_member_4a0a7f3e = R"DOC(Block index of the operator.)DOC";
constexpr auto _c2py_doc_member_4a0c2753 = R"DOC(Inner index within the block.)DOC";
constexpr auto _c2py_doc_member_f8752f46 = R"DOC(Whether the operator is a dagger (creation operator).)DOC";
constexpr auto _c2py_doc_member_8b8a6756 = R"DOC(Cumulative (linear) index.)DOC";
static PyObject *prop_get_dict_7e5d6987(PyObject *self, void *) {
  auto &self_c = *(((c2py::wrap<_c2py_cls_6805ae04> *)self)->_c);
  c2py::pydict dic;
  dic["block_index"]  = self_c.block_index;
  dic["inner_index"]  = self_c.inner_index;
  dic["dagger"]       = self_c.dagger;
  dic["linear_index"] = self_c.linear_index;
  return dic.new_ref();
}

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_6805ae04>[] = {
   c2py::getsetdef_from_member<&_c2py_cls_6805ae04::block_index, _c2py_cls_6805ae04>("block_index", _c2py_doc_member_4a0a7f3e),
   c2py::getsetdef_from_member<&_c2py_cls_6805ae04::inner_index, _c2py_cls_6805ae04>("inner_index", _c2py_doc_member_4a0c2753),
   c2py::getsetdef_from_member<&_c2py_cls_6805ae04::dagger, _c2py_cls_6805ae04>("dagger", _c2py_doc_member_f8752f46),
   c2py::getsetdef_from_member<&_c2py_cls_6805ae04::linear_index, _c2py_cls_6805ae04>("linear_index", _c2py_doc_member_8b8a6756),
   {"__dict__", (getter)prop_get_dict_7e5d6987, nullptr, "", nullptr},
   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<_c2py_cls_6805ae04> =
   R"DOC(Description of a creation/annihilation operator.)DOC" + std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<_c2py_cls_6805ae04>;
// --------- class _c2py_cls_828b6936 -----------
using _c2py_cls_828b6936                                            = triqs_cthyb::op_desc_pair_t;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_828b6936>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_828b6936> = "triqs_cthyb.configuration.OpDescPairT";

static int synth_constructor_a1aecb80(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (args and PyTuple_Check(args) and (PyTuple_Size(args) > 0)) {
    PyErr_SetString(PyExc_RuntimeError,
                    ("Error in constructing triqs_cthyb::op_desc_pair_t.\nNo positional arguments allowed. Use keywords arguments"));
    return -1;
  }
  c2py::pydict_extractor de{kwargs};
  try {
    ((c2py::wrap<_c2py_cls_828b6936> *)self)->_c = new _c2py_cls_828b6936{};
  } catch (std::exception const &e) {
    PyErr_SetString(PyExc_RuntimeError, ("Error in constructing triqs_cthyb::op_desc_pair_t from a Python dict.\n   "s + e.what()).c_str());
    return -1;
  }
  auto &self_c = *(((c2py::wrap<_c2py_cls_828b6936> *)self)->_c);
  de("opL", self_c.opL, false);
  de("opR", self_c.opR, false);
  return de.check();
}

template <> constexpr initproc c2py::tp_init<_c2py_cls_828b6936> = synth_constructor_a1aecb80;

template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_828b6936> =
   c2py::replace_tags(R"DOC(Synthesized constructor with the following keyword arguments:

Parameters
----------
opL : {par_0}

opR : {par_1}

)DOC",
                      "par", {c2py::python_typename<triqs_cthyb::op_desc>(), c2py::python_typename<triqs_cthyb::op_desc>()});

// ----- Method table ----
// clang-format off
template <> PyMethodDef c2py::tp_methods<_c2py_cls_828b6936>[] = {
   {"__write_hdf5__", c2py::tpxx_write_h5<_c2py_cls_828b6936>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_h5<_c2py_cls_828b6936>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_h5<_c2py_cls_828b6936>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};
// clang-format on

constexpr auto _c2py_doc_member_8c2e3773 = R"DOC()DOC";
constexpr auto _c2py_doc_member_aa2e66ad = R"DOC()DOC";
static PyObject *prop_get_dict_a1aecb80(PyObject *self, void *) {
  auto &self_c = *(((c2py::wrap<_c2py_cls_828b6936> *)self)->_c);
  c2py::pydict dic;
  dic["opL"] = self_c.opL;
  dic["opR"] = self_c.opR;
  return dic.new_ref();
}

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_828b6936>[] = {
   c2py::getsetdef_from_member<&_c2py_cls_828b6936::opL, _c2py_cls_828b6936>("opL", _c2py_doc_member_8c2e3773),
   c2py::getsetdef_from_member<&_c2py_cls_828b6936::opR, _c2py_cls_828b6936>("opR", _c2py_doc_member_aa2e66ad),
   {"__dict__", (getter)prop_get_dict_a1aecb80, nullptr, "", nullptr},
   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <> const std::string c2py::tp_doc<_c2py_cls_828b6936> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_828b6936>;
// --------- class _c2py_cls_bb185c2c -----------
using _c2py_cls_bb185c2c                                            = triqs_cthyb::bosonic_op_pair_t;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_bb185c2c>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_bb185c2c> = "triqs_cthyb.configuration.BosonicOpPairT";

static int synth_constructor_6a33906e(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (args and PyTuple_Check(args) and (PyTuple_Size(args) > 0)) {
    PyErr_SetString(PyExc_RuntimeError,
                    ("Error in constructing triqs_cthyb::bosonic_op_pair_t.\nNo positional arguments allowed. Use keywords arguments"));
    return -1;
  }
  c2py::pydict_extractor de{kwargs};
  try {
    ((c2py::wrap<_c2py_cls_bb185c2c> *)self)->_c = new _c2py_cls_bb185c2c{};
  } catch (std::exception const &e) {
    PyErr_SetString(PyExc_RuntimeError, ("Error in constructing triqs_cthyb::bosonic_op_pair_t from a Python dict.\n   "s + e.what()).c_str());
    return -1;
  }
  auto &self_c = *(((c2py::wrap<_c2py_cls_bb185c2c> *)self)->_c);
  de("op1", self_c.op1, false);
  de("op2", self_c.op2, false);
  de("f_index", self_c.f_index, false);
  return de.check();
}

template <> constexpr initproc c2py::tp_init<_c2py_cls_bb185c2c> = synth_constructor_6a33906e;

template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_bb185c2c> = c2py::replace_tags(
   R"DOC(Synthesized constructor with the following keyword arguments:

Parameters
----------
op1 : {par_0}

op2 : {par_1}

f_index : {par_2}

)DOC",
   "par", {c2py::python_typename<triqs_cthyb::op_desc_pair_t>(), c2py::python_typename<triqs_cthyb::op_desc_pair_t>(), c2py::python_typename<int>()});

// ----- Method table ----
// clang-format off
template <> PyMethodDef c2py::tp_methods<_c2py_cls_bb185c2c>[] = {
   {"__write_hdf5__", c2py::tpxx_write_h5<_c2py_cls_bb185c2c>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_h5<_c2py_cls_bb185c2c>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_h5<_c2py_cls_bb185c2c>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};
// clang-format on

constexpr auto _c2py_doc_member_6c34b49a = R"DOC()DOC";
constexpr auto _c2py_doc_member_6b34b307 = R"DOC()DOC";
constexpr auto _c2py_doc_member_bc45c635 = R"DOC()DOC";
static PyObject *prop_get_dict_6a33906e(PyObject *self, void *) {
  auto &self_c = *(((c2py::wrap<_c2py_cls_bb185c2c> *)self)->_c);
  c2py::pydict dic;
  dic["op1"]     = self_c.op1;
  dic["op2"]     = self_c.op2;
  dic["f_index"] = self_c.f_index;
  return dic.new_ref();
}

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_bb185c2c>[] = {
   c2py::getsetdef_from_member<&_c2py_cls_bb185c2c::op1, _c2py_cls_bb185c2c>("op1", _c2py_doc_member_6c34b49a),
   c2py::getsetdef_from_member<&_c2py_cls_bb185c2c::op2, _c2py_cls_bb185c2c>("op2", _c2py_doc_member_6b34b307),
   c2py::getsetdef_from_member<&_c2py_cls_bb185c2c::f_index, _c2py_cls_bb185c2c>("f_index", _c2py_doc_member_bc45c635),
   {"__dict__", (getter)prop_get_dict_6a33906e, nullptr, "", nullptr},
   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <> const std::string c2py::tp_doc<_c2py_cls_bb185c2c> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_bb185c2c>;
// --------- class _c2py_cls_5e49151b -----------
using _c2py_cls_5e49151b                                            = triqs_cthyb::configuration;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_5e49151b>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_5e49151b> = "triqs_cthyb.configuration.Configuration";
static const auto _c2py_init_62c50a24 = c2py::dispatcher_c_kw_t{c2py::c_constructor<_c2py_cls_5e49151b, double, long>("beta", "id"_a = 0)};
template <> constexpr initproc c2py::tp_init<_c2py_cls_5e49151b>    = c2py::pyfkw_constructor<_c2py_init_62c50a24>;
template <> const std::string c2py::tp_ctor_doc<_c2py_cls_5e49151b> = _c2py_init_62c50a24.doc(R"DOC()DOC");
// clear
static auto const _c2py_fun_a52e0d94 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_5e49151b &self) -> decltype(auto) { return self.clear(); }, "self")};

// erase
static auto const _c2py_fun_8b2b8697 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](_c2py_cls_5e49151b &self, const triqs::utility::time_pt &t) -> decltype(auto) { return self.erase(t); }, "self", "t")};

// insert
static auto const _c2py_fun_b3a77f9a = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_5e49151b &self, triqs::utility::time_pt tau, triqs_cthyb::op_desc op) -> decltype(auto) { return self.insert(tau, op); }, "self",
   "tau", "op")};

// replace
static auto const _c2py_fun_48eed345 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_5e49151b &self, triqs::utility::time_pt tau, triqs_cthyb::op_desc op) -> decltype(auto) { return self.replace(tau, op); }, "self",
   "tau", "op")};

static const auto _c2py_doc_a52e0d94 = _c2py_fun_a52e0d94.doc(R"DOC(
Clear the configuration (remove all operators).
)DOC");
static const auto _c2py_doc_8b2b8697 = _c2py_fun_8b2b8697.doc(R"DOC(
Erase the operator at a given imaginary time.

Parameters
----------
tau : {par_0}
   Imaginary time at which to erase the operator.
)DOC",
                                                              {{}});
static const auto _c2py_doc_b3a77f9a =
   _c2py_fun_b3a77f9a.doc(R"DOC(
Insert a given operator at a given imaginary time.

Parameters
----------
tau : {par_0}
   Imaginary time at which to insert the operator.
op : {par_1}
   Description of the operator to insert.
)DOC",
                          {{c2py::python_typename<triqs::utility::time_pt>()}, {c2py::python_typename<triqs_cthyb::op_desc>()}});
static const auto _c2py_doc_48eed345 =
   _c2py_fun_48eed345.doc(R"DOC(
Replace an existing operator at a given imaginary time with a new one.

Parameters
----------
tau : {par_0}
   Imaginary time at which to replace the operator.
op : {par_1}
   Description of the operator to insert.
)DOC",
                          {{c2py::python_typename<triqs::utility::time_pt>()}, {c2py::python_typename<triqs_cthyb::op_desc>()}});

// ----- Method table ----
// clang-format off
template <> PyMethodDef c2py::tp_methods<_c2py_cls_5e49151b>[] = {
   PMDF("clear", a52e0d94),
   PMDF("erase", 8b2b8697),
   PMDF("insert", b3a77f9a),
   PMDF("replace", 48eed345),
   {"__write_hdf5__", c2py::tpxx_write_h5<_c2py_cls_5e49151b>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_h5<_c2py_cls_5e49151b>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_h5<_c2py_cls_5e49151b>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};
// clang-format on

static constexpr auto prop_doc_a278d8a1 = R"DOC(Inverse temperature :math:`\beta`.)DOC";

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_5e49151b>[] = {

   {"beta", c2py::getter_from_method<c2py::castmc<>(&triqs_cthyb::configuration::beta)>, nullptr, prop_doc_a278d8a1, nullptr},
   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <> PyMappingMethods c2py::tp_as_mapping<_c2py_cls_5e49151b> = {c2py::tpxx_size<_c2py_cls_5e49151b>, nullptr, nullptr};

template <>
const std::string c2py::tp_doc<_c2py_cls_5e49151b> = R"DOC(Configuration of the Monte Carlo simulation (operators on the imaginary-time line).)DOC"
   + std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<_c2py_cls_5e49151b>;
// --------- class _c2py_cls_d8db9312 -----------
using _c2py_cls_d8db9312                                            = triqs_cthyb::configuration::dyn_bosonic_pair_t;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_d8db9312>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_d8db9312> = "triqs_cthyb.configuration.DynBosonicPairT";

static int synth_constructor_3d0bd9e1(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (args and PyTuple_Check(args) and (PyTuple_Size(args) > 0)) {
    PyErr_SetString(
       PyExc_RuntimeError,
       ("Error in constructing triqs_cthyb::configuration::dyn_bosonic_pair_t.\nNo positional arguments allowed. Use keywords arguments"));
    return -1;
  }
  c2py::pydict_extractor de{kwargs};
  try {
    ((c2py::wrap<_c2py_cls_d8db9312> *)self)->_c = new _c2py_cls_d8db9312{};
  } catch (std::exception const &e) {
    PyErr_SetString(PyExc_RuntimeError,
                    ("Error in constructing triqs_cthyb::configuration::dyn_bosonic_pair_t from a Python dict.\n   "s + e.what()).c_str());
    return -1;
  }
  auto &self_c = *(((c2py::wrap<_c2py_cls_d8db9312> *)self)->_c);
  de("ops", self_c.ops, false);
  de("tau1", self_c.tau1, false);
  de("tau2", self_c.tau2, false);
  return de.check();
}

template <> constexpr initproc c2py::tp_init<_c2py_cls_d8db9312> = synth_constructor_3d0bd9e1;

template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_d8db9312> =
   c2py::replace_tags(R"DOC(Synthesized constructor with the following keyword arguments:

Parameters
----------
ops : {par_0}

tau1 : {par_1}

tau2 : {par_2}

)DOC",
                      "par",
                      {c2py::python_typename<triqs_cthyb::bosonic_op_pair_t>(), c2py::python_typename<triqs::utility::time_pt>(),
                       c2py::python_typename<triqs::utility::time_pt>()});

// ----- Method table ----
// clang-format off
template <> PyMethodDef c2py::tp_methods<_c2py_cls_d8db9312>[] = {
   {"__write_hdf5__", c2py::tpxx_write_h5<_c2py_cls_d8db9312>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_h5<_c2py_cls_d8db9312>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_h5<_c2py_cls_d8db9312>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};
// clang-format on

constexpr auto _c2py_doc_member_527b9bb6 = R"DOC()DOC";
constexpr auto _c2py_doc_member_44d1f443 = R"DOC()DOC";
constexpr auto _c2py_doc_member_45d1f5d6 = R"DOC()DOC";
static PyObject *prop_get_dict_3d0bd9e1(PyObject *self, void *) {
  auto &self_c = *(((c2py::wrap<_c2py_cls_d8db9312> *)self)->_c);
  c2py::pydict dic;
  dic["ops"]  = self_c.ops;
  dic["tau1"] = self_c.tau1;
  dic["tau2"] = self_c.tau2;
  return dic.new_ref();
}

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_d8db9312>[] = {
   c2py::getsetdef_from_member<&_c2py_cls_d8db9312::ops, _c2py_cls_d8db9312>("ops", _c2py_doc_member_527b9bb6),
   c2py::getsetdef_from_member<&_c2py_cls_d8db9312::tau1, _c2py_cls_d8db9312>("tau1", _c2py_doc_member_44d1f443),
   c2py::getsetdef_from_member<&_c2py_cls_d8db9312::tau2, _c2py_cls_d8db9312>("tau2", _c2py_doc_member_45d1f5d6),
   {"__dict__", (getter)prop_get_dict_3d0bd9e1, nullptr, "", nullptr},
   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <> const std::string c2py::tp_doc<_c2py_cls_d8db9312> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_d8db9312>;

// ==================== module functions ====================

//--------------------- module function table  -----------------------------

// clang-format off
static PyMethodDef module_methods[] = {
   {nullptr, nullptr, 0, nullptr}  // Sentinel
};
// clang-format on

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "configuration",                                    /* name of module */
                                        R"RAWDOC(MC configuration for triqs_cthyb.)RAWDOC", /* module documentation, may be NULL */
                                        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
                                        module_methods,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit_configuration() {

  if (not c2py::check_python_version("configuration")) return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_6805ae04>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_828b6936>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_bb185c2c>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_5e49151b>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_d8db9312>) < 0) return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  if (not c2py::register_internal_types()) return NULL;
#define _add_type(T, N)                                                                                                                              \
  if (not c2py::add_type_object_to_main<T>(N, m)) return NULL
  _add_type(_c2py_cls_6805ae04, "OpDesc");
  _add_type(_c2py_cls_828b6936, "OpDescPairT");
  _add_type(_c2py_cls_bb185c2c, "BosonicOpPairT");
  _add_type(_c2py_cls_5e49151b, "Configuration");
  _add_type(_c2py_cls_d8db9312, "DynBosonicPairT");
#undef _add_type

  c2py::pyref module = c2py::pyref::module("h5.formats");
  if (not module) return nullptr;
  c2py::pyref register_class = module.attr("register_class");

  register_h5_type<_c2py_cls_6805ae04>(register_class);
  register_h5_type<_c2py_cls_828b6936>(register_class);
  register_h5_type<_c2py_cls_bb185c2c>(register_class);
  register_h5_type<_c2py_cls_5e49151b>(register_class);
  register_h5_type<_c2py_cls_d8db9312>(register_class);

  return m;
}
#endif
// CLAIR_WRAP_GEN
