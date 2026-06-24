
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

#define C2PY_VERSION_MAJOR 0
#define C2PY_VERSION_MINOR 1

#include <c2py/c2py.hpp>
#include <c2py/serialization/h5.hpp>

using c2py::operator""_a;

// ==================== enums =====================

// ==================== module classes =====================

// --------- class _c2py_cls_0 -----------
using _c2py_cls_0                                            = triqs_cthyb::op_desc;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_0>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_0> = "triqs_cthyb.configuration.OpDesc";

static int synth_constructor_0(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (args and PyTuple_Check(args) and (PyTuple_Size(args) > 0)) {
    PyErr_SetString(PyExc_RuntimeError, ("Error in constructing triqs_cthyb::op_desc.\nNo positional arguments allowed. Use keywords arguments"));
    return -1;
  }
  c2py::pydict_extractor de{kwargs};
  try {
    ((c2py::wrap<_c2py_cls_0> *)self)->_c = new _c2py_cls_0{};
  } catch (std::exception const &e) {
    PyErr_SetString(PyExc_RuntimeError, ("Error in constructing triqs_cthyb::op_desc from a Python dict.\n   "s + e.what()).c_str());
    return -1;
  }
  auto &self_c = *(((c2py::wrap<_c2py_cls_0> *)self)->_c);
  de("block_index", self_c.block_index, false);
  de("inner_index", self_c.inner_index, false);
  de("dagger", self_c.dagger, false);
  de("linear_index", self_c.linear_index, false);
  return de.check();
}

template <> constexpr initproc c2py::tp_init<_c2py_cls_0> = synth_constructor_0;

template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_0> =
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
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_0>[] = {
   {"__write_hdf5__", c2py::tpxx_write_h5<_c2py_cls_0>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_h5<_c2py_cls_0>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_h5<_c2py_cls_0>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

constexpr auto _c2py_doc_member_0 = R"DOC(Block index of the operator.)DOC";
constexpr auto _c2py_doc_member_1 = R"DOC(Inner index within the block.)DOC";
constexpr auto _c2py_doc_member_2 = R"DOC(Whether the operator is a dagger (creation operator).)DOC";
constexpr auto _c2py_doc_member_3 = R"DOC(Cumulative (linear) index.)DOC";
static PyObject *prop_get_dict_0(PyObject *self, void *) {
  auto &self_c = *(((c2py::wrap<_c2py_cls_0> *)self)->_c);
  c2py::pydict dic;
  dic["block_index"]  = self_c.block_index;
  dic["inner_index"]  = self_c.inner_index;
  dic["dagger"]       = self_c.dagger;
  dic["linear_index"] = self_c.linear_index;
  return dic.new_ref();
}

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_0>[] = {
   c2py::getsetdef_from_member<&_c2py_cls_0::block_index, _c2py_cls_0>("block_index", _c2py_doc_member_0),
   c2py::getsetdef_from_member<&_c2py_cls_0::inner_index, _c2py_cls_0>("inner_index", _c2py_doc_member_1),
   c2py::getsetdef_from_member<&_c2py_cls_0::dagger, _c2py_cls_0>("dagger", _c2py_doc_member_2),
   c2py::getsetdef_from_member<&_c2py_cls_0::linear_index, _c2py_cls_0>("linear_index", _c2py_doc_member_3),
   {"__dict__", (getter)prop_get_dict_0, nullptr, "", nullptr},
   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<_c2py_cls_0> =
   R"DOC(Description of a creation/annihilation operator.)DOC" + std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<_c2py_cls_0>;
// --------- class _c2py_cls_1 -----------
using _c2py_cls_1                                            = triqs_cthyb::op_desc_pair_t;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_1>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_1> = "triqs_cthyb.configuration.OpDescPairT";

static int synth_constructor_1(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (args and PyTuple_Check(args) and (PyTuple_Size(args) > 0)) {
    PyErr_SetString(PyExc_RuntimeError,
                    ("Error in constructing triqs_cthyb::op_desc_pair_t.\nNo positional arguments allowed. Use keywords arguments"));
    return -1;
  }
  c2py::pydict_extractor de{kwargs};
  try {
    ((c2py::wrap<_c2py_cls_1> *)self)->_c = new _c2py_cls_1{};
  } catch (std::exception const &e) {
    PyErr_SetString(PyExc_RuntimeError, ("Error in constructing triqs_cthyb::op_desc_pair_t from a Python dict.\n   "s + e.what()).c_str());
    return -1;
  }
  auto &self_c = *(((c2py::wrap<_c2py_cls_1> *)self)->_c);
  de("opL", self_c.opL, false);
  de("opR", self_c.opR, false);
  return de.check();
}

template <> constexpr initproc c2py::tp_init<_c2py_cls_1> = synth_constructor_1;

template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_1> =
   c2py::replace_tags(R"DOC(Synthesized constructor with the following keyword arguments:

Parameters
----------
opL : {par_0}

opR : {par_1}

)DOC",
                      "par", {c2py::python_typename<triqs_cthyb::op_desc>(), c2py::python_typename<triqs_cthyb::op_desc>()});

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_1>[] = {
   {"__write_hdf5__", c2py::tpxx_write_h5<_c2py_cls_1>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_h5<_c2py_cls_1>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_h5<_c2py_cls_1>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

constexpr auto _c2py_doc_member_4 = R"DOC()DOC";
constexpr auto _c2py_doc_member_5 = R"DOC()DOC";
static PyObject *prop_get_dict_1(PyObject *self, void *) {
  auto &self_c = *(((c2py::wrap<_c2py_cls_1> *)self)->_c);
  c2py::pydict dic;
  dic["opL"] = self_c.opL;
  dic["opR"] = self_c.opR;
  return dic.new_ref();
}

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_1>[] = {c2py::getsetdef_from_member<&_c2py_cls_1::opL, _c2py_cls_1>("opL", _c2py_doc_member_4),
                                                        c2py::getsetdef_from_member<&_c2py_cls_1::opR, _c2py_cls_1>("opR", _c2py_doc_member_5),
                                                        {"__dict__", (getter)prop_get_dict_1, nullptr, "", nullptr},
                                                        {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <> const std::string c2py::tp_doc<_c2py_cls_1> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_1>;
// --------- class _c2py_cls_2 -----------
using _c2py_cls_2                                            = triqs_cthyb::bosonic_op_pair_t;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_2>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_2> = "triqs_cthyb.configuration.BosonicOpPairT";

static int synth_constructor_2(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (args and PyTuple_Check(args) and (PyTuple_Size(args) > 0)) {
    PyErr_SetString(PyExc_RuntimeError,
                    ("Error in constructing triqs_cthyb::bosonic_op_pair_t.\nNo positional arguments allowed. Use keywords arguments"));
    return -1;
  }
  c2py::pydict_extractor de{kwargs};
  try {
    ((c2py::wrap<_c2py_cls_2> *)self)->_c = new _c2py_cls_2{};
  } catch (std::exception const &e) {
    PyErr_SetString(PyExc_RuntimeError, ("Error in constructing triqs_cthyb::bosonic_op_pair_t from a Python dict.\n   "s + e.what()).c_str());
    return -1;
  }
  auto &self_c = *(((c2py::wrap<_c2py_cls_2> *)self)->_c);
  de("op1", self_c.op1, false);
  de("op2", self_c.op2, false);
  de("f_index", self_c.f_index, false);
  return de.check();
}

template <> constexpr initproc c2py::tp_init<_c2py_cls_2> = synth_constructor_2;

template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_2> = c2py::replace_tags(
   R"DOC(Synthesized constructor with the following keyword arguments:

Parameters
----------
op1 : {par_0}

op2 : {par_1}

f_index : {par_2}

)DOC",
   "par", {c2py::python_typename<triqs_cthyb::op_desc_pair_t>(), c2py::python_typename<triqs_cthyb::op_desc_pair_t>(), c2py::python_typename<int>()});

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_2>[] = {
   {"__write_hdf5__", c2py::tpxx_write_h5<_c2py_cls_2>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_h5<_c2py_cls_2>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_h5<_c2py_cls_2>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

constexpr auto _c2py_doc_member_6 = R"DOC()DOC";
constexpr auto _c2py_doc_member_7 = R"DOC()DOC";
constexpr auto _c2py_doc_member_8 = R"DOC()DOC";
static PyObject *prop_get_dict_2(PyObject *self, void *) {
  auto &self_c = *(((c2py::wrap<_c2py_cls_2> *)self)->_c);
  c2py::pydict dic;
  dic["op1"]     = self_c.op1;
  dic["op2"]     = self_c.op2;
  dic["f_index"] = self_c.f_index;
  return dic.new_ref();
}

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_2>[] = {
   c2py::getsetdef_from_member<&_c2py_cls_2::op1, _c2py_cls_2>("op1", _c2py_doc_member_6),
   c2py::getsetdef_from_member<&_c2py_cls_2::op2, _c2py_cls_2>("op2", _c2py_doc_member_7),
   c2py::getsetdef_from_member<&_c2py_cls_2::f_index, _c2py_cls_2>("f_index", _c2py_doc_member_8),
   {"__dict__", (getter)prop_get_dict_2, nullptr, "", nullptr},
   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <> const std::string c2py::tp_doc<_c2py_cls_2> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_2>;
// --------- class _c2py_cls_3 -----------
using _c2py_cls_3                                            = triqs_cthyb::configuration;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_3>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_3> = "triqs_cthyb.configuration.Configuration";
static const auto _c2py_init_0 = c2py::dispatcher_c_kw_t{c2py::c_constructor<_c2py_cls_3, double, long>("beta", "id"_a = 0)};
template <> constexpr initproc c2py::tp_init<_c2py_cls_3>    = c2py::pyfkw_constructor<_c2py_init_0>;
template <> const std::string c2py::tp_ctor_doc<_c2py_cls_3> = _c2py_init_0.doc(R"DOC()DOC");
// clear
static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_3 &self) -> decltype(auto) { return self.clear(); }, "self")};

// erase
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](_c2py_cls_3 &self, const triqs::utility::time_pt &t) -> decltype(auto) { return self.erase(t); }, "self", "t")};

// insert
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](_c2py_cls_3 &self, triqs::utility::time_pt tau, triqs_cthyb::op_desc op) -> decltype(auto) { return self.insert(tau, op); },
                 "self", "tau", "op")};

// replace
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](_c2py_cls_3 &self, triqs::utility::time_pt tau, triqs_cthyb::op_desc op) -> decltype(auto) { return self.replace(tau, op); },
                 "self", "tau", "op")};

static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC(
Clear the configuration (remove all operators).
)DOC");
static const auto _c2py_doc_1 = _c2py_fun_1.doc(R"DOC(
Erase the operator at a given imaginary time.

Parameters
----------
tau : {par_0}
   Imaginary time at which to erase the operator.
)DOC",
                                                {{}});
static const auto _c2py_doc_2 =
   _c2py_fun_2.doc(R"DOC(
Insert a given operator at a given imaginary time.

Parameters
----------
tau : {par_0}
   Imaginary time at which to insert the operator.
op : {par_1}
   Description of the operator to insert.
)DOC",
                   {{c2py::python_typename<triqs::utility::time_pt>()}, {c2py::python_typename<triqs_cthyb::op_desc>()}});
static const auto _c2py_doc_3 =
   _c2py_fun_3.doc(R"DOC(
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
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_3>[] = {
   {"clear", (PyCFunction)c2py::pyfkw<_c2py_fun_0>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
   {"erase", (PyCFunction)c2py::pyfkw<_c2py_fun_1>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
   {"insert", (PyCFunction)c2py::pyfkw<_c2py_fun_2>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
   {"replace", (PyCFunction)c2py::pyfkw<_c2py_fun_3>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
   {"__write_hdf5__", c2py::tpxx_write_h5<_c2py_cls_3>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_h5<_c2py_cls_3>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_h5<_c2py_cls_3>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

static constexpr auto prop_doc_0 = R"DOC(Inverse temperature :math:`\beta`.)DOC";

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_3>[] = {

   {"beta", c2py::getter_from_method<c2py::castmc<>(&triqs_cthyb::configuration::beta)>, nullptr, prop_doc_0, nullptr},
   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <> PyMappingMethods c2py::tp_as_mapping<_c2py_cls_3> = {c2py::tpxx_size<_c2py_cls_3>, nullptr, nullptr};

template <>
const std::string c2py::tp_doc<_c2py_cls_3> = R"DOC(Configuration of the Monte Carlo simulation (operators on the imaginary-time line).)DOC"
   + std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<_c2py_cls_3>;
// --------- class _c2py_cls_4 -----------
using _c2py_cls_4                                            = triqs_cthyb::configuration::dyn_bosonic_pair_t;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_4>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_4> = "triqs_cthyb.configuration.DynBosonicPairT";

static int synth_constructor_3(PyObject *self, PyObject *args, PyObject *kwargs) {
  if (args and PyTuple_Check(args) and (PyTuple_Size(args) > 0)) {
    PyErr_SetString(
       PyExc_RuntimeError,
       ("Error in constructing triqs_cthyb::configuration::dyn_bosonic_pair_t.\nNo positional arguments allowed. Use keywords arguments"));
    return -1;
  }
  c2py::pydict_extractor de{kwargs};
  try {
    ((c2py::wrap<_c2py_cls_4> *)self)->_c = new _c2py_cls_4{};
  } catch (std::exception const &e) {
    PyErr_SetString(PyExc_RuntimeError,
                    ("Error in constructing triqs_cthyb::configuration::dyn_bosonic_pair_t from a Python dict.\n   "s + e.what()).c_str());
    return -1;
  }
  auto &self_c = *(((c2py::wrap<_c2py_cls_4> *)self)->_c);
  de("ops", self_c.ops, false);
  de("tau1", self_c.tau1, false);
  de("tau2", self_c.tau2, false);
  return de.check();
}

template <> constexpr initproc c2py::tp_init<_c2py_cls_4> = synth_constructor_3;

template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_4> =
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
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_4>[] = {
   {"__write_hdf5__", c2py::tpxx_write_h5<_c2py_cls_4>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_h5<_c2py_cls_4>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_h5<_c2py_cls_4>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

constexpr auto _c2py_doc_member_9  = R"DOC()DOC";
constexpr auto _c2py_doc_member_10 = R"DOC()DOC";
constexpr auto _c2py_doc_member_11 = R"DOC()DOC";
static PyObject *prop_get_dict_3(PyObject *self, void *) {
  auto &self_c = *(((c2py::wrap<_c2py_cls_4> *)self)->_c);
  c2py::pydict dic;
  dic["ops"]  = self_c.ops;
  dic["tau1"] = self_c.tau1;
  dic["tau2"] = self_c.tau2;
  return dic.new_ref();
}

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_4>[] = {c2py::getsetdef_from_member<&_c2py_cls_4::ops, _c2py_cls_4>("ops", _c2py_doc_member_9),
                                                        c2py::getsetdef_from_member<&_c2py_cls_4::tau1, _c2py_cls_4>("tau1", _c2py_doc_member_10),
                                                        c2py::getsetdef_from_member<&_c2py_cls_4::tau2, _c2py_cls_4>("tau2", _c2py_doc_member_11),
                                                        {"__dict__", (getter)prop_get_dict_3, nullptr, "", nullptr},
                                                        {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <> const std::string c2py::tp_doc<_c2py_cls_4> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_4>;

// ==================== module functions ====================

//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

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

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_0>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_1>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_2>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_3>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_4>) < 0) return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;
#define _add_type(T, N) c2py::add_type_object_to_main<T>(N, m, conv_table)
  _add_type(_c2py_cls_0, "OpDesc");
  _add_type(_c2py_cls_1, "OpDescPairT");
  _add_type(_c2py_cls_2, "BosonicOpPairT");
  _add_type(_c2py_cls_3, "Configuration");
  _add_type(_c2py_cls_4, "DynBosonicPairT");
#undef _add_type

  c2py::pyref module = c2py::pyref::module("h5.formats");
  if (not module) return nullptr;
  c2py::pyref register_class = module.attr("register_class");

  register_h5_type<_c2py_cls_0>(register_class);
  register_h5_type<_c2py_cls_1>(register_class);
  register_h5_type<_c2py_cls_2>(register_class);
  register_h5_type<_c2py_cls_3>(register_class);
  register_h5_type<_c2py_cls_4>(register_class);

  return m;
}
#endif
// CLAIR_WRAP_GEN
