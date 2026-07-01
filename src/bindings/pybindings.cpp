#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "physics/Memristor.h"
#include "physics/Crossbar.h"

namespace py = pybind11;

PYBIND11_MODULE(memristorsim, m) {
    m.doc() = "Memristor 3D Simulator Python Bindings";

    // Bind ConductionModel
    py::enum_<ConductionModel>(m, "ConductionModel")
        .value("Sinh", ConductionModel::Sinh)
        .value("PooleFrenkel", ConductionModel::PooleFrenkel)
        .value("Schottky", ConductionModel::Schottky)
        .export_values();

    // Bind MemristorParams
    py::class_<MemristorParams>(m, "MemristorParams")
        .def(py::init<>())
        .def_readwrite("R_on", &MemristorParams::R_on)
        .def_readwrite("R_off", &MemristorParams::R_off)
        .def_readwrite("v_on", &MemristorParams::v_on)
        .def_readwrite("v_off", &MemristorParams::v_off)
        .def_readwrite("k_on", &MemristorParams::k_on)
        .def_readwrite("k_off", &MemristorParams::k_off)
        .def_readwrite("alpha_on", &MemristorParams::alpha_on)
        .def_readwrite("alpha_off", &MemristorParams::alpha_off)
        .def_readwrite("w_init", &MemristorParams::w_init)
        .def_readwrite("I_compliance", &MemristorParams::I_compliance)
        .def_readwrite("theta_thermal", &MemristorParams::theta_thermal)
        .def_readwrite("T_critical", &MemristorParams::T_critical)
        .def_readwrite("conduction_model", &MemristorParams::conduction_model)
        .def_readwrite("gamma_sinh", &MemristorParams::gamma_sinh)
        .def_readwrite("beta_pf", &MemristorParams::beta_pf)
        .def_readwrite("beta_sc", &MemristorParams::beta_sc)
        .def_readwrite("enable_variability", &MemristorParams::enable_variability)
        .def_readwrite("sigma_w_init", &MemristorParams::sigma_w_init)
        .def_readwrite("sigma_k_on", &MemristorParams::sigma_k_on)
        .def_readwrite("sigma_c2c", &MemristorParams::sigma_c2c)
        .def_readwrite("enable_rtn", &MemristorParams::enable_rtn)
        .def_readwrite("rtn_amplitude", &MemristorParams::rtn_amplitude)
        .def_readwrite("rtn_tau_c", &MemristorParams::rtn_tau_c)
        .def_readwrite("rtn_tau_e", &MemristorParams::rtn_tau_e)
        .def_readwrite("enable_selector", &MemristorParams::enable_selector)
        .def_readwrite("selector_type", &MemristorParams::selector_type)
        .def_readwrite("selector_v_th", &MemristorParams::selector_v_th)
        .def_readwrite("selector_alpha", &MemristorParams::selector_alpha)
        .def_readwrite("selector_v_gate", &MemristorParams::selector_v_gate)
        .def_readwrite("selector_v_th_trans", &MemristorParams::selector_v_th_trans);

    // Bind PhysicsEngine
    py::class_<PhysicsEngine>(m, "PhysicsEngine")
        .def(py::init<const MemristorParams&>())
        .def("reset", &PhysicsEngine::reset)
        .def("update", &PhysicsEngine::update)
        .def("w", &PhysicsEngine::w)
        .def("r", &PhysicsEngine::r)
        .def("i", &PhysicsEngine::i)
        .def("power", &PhysicsEngine::power)
        .def("dT", &PhysicsEngine::dT)
        .def("calculate_current", &PhysicsEngine::calculate_current)
        .def("calculate_memristor_current", &PhysicsEngine::calculate_memristor_current)
        .def("calculate_selector_current", &PhysicsEngine::calculate_selector_current)
        .def("params", &PhysicsEngine::params, py::return_value_policy::reference_internal)
        .def("set_params", &PhysicsEngine::set_params)
        .def("set_w", &PhysicsEngine::set_w);

    // Bind CrossbarArray
    py::class_<CrossbarArray>(m, "CrossbarArray")
        .def(py::init<>())
        .def("reset", &CrossbarArray::reset)
        .def("set_inputs", &CrossbarArray::set_inputs)
        .def("inputs", &CrossbarArray::inputs)
        .def("outputs", &CrossbarArray::outputs)
        .def("w", &CrossbarArray::w)
        .def("r", &CrossbarArray::r)
        .def("power", &CrossbarArray::power)
        .def("dT", &CrossbarArray::dT)
        .def("get_device", &CrossbarArray::get_device, py::return_value_policy::reference_internal)
        .def("set_params", &CrossbarArray::set_params)
        .def("enable_ir_drop", &CrossbarArray::enable_ir_drop)
        .def("set_enable_ir_drop", &CrossbarArray::set_enable_ir_drop)
        .def("r_wire", &CrossbarArray::r_wire)
        .def("set_r_wire", &CrossbarArray::set_r_wire)
        .def("v_row_node", &CrossbarArray::v_row_node)
        .def("v_col_node", &CrossbarArray::v_col_node)
        .def("enable_dac", &CrossbarArray::enable_dac)
        .def("set_enable_dac", &CrossbarArray::set_enable_dac)
        .def("dac_bits", &CrossbarArray::dac_bits)
        .def("set_dac_bits", &CrossbarArray::set_dac_bits)
        .def("enable_adc", &CrossbarArray::enable_adc)
        .def("set_enable_adc", &CrossbarArray::set_enable_adc)
        .def("adc_bits", &CrossbarArray::adc_bits)
        .def("set_adc_bits", &CrossbarArray::set_adc_bits)
        .def("update", &CrossbarArray::update)
        .def("program_cell", &CrossbarArray::program_cell);
}
