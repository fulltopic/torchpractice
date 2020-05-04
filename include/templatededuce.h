#include <ATen/core/op_registration/infer_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <iostream>

namespace c10 {

using Stack = torch::jit::Stack; // TODO Instead of this, move torch::jit::Stack to the c10 namespace.

/**
 * Inherit from OperatorKernel to implement a c10 kernel.
 *
 * Example:
 * > namespace {
 * >   class my_kernel_cpu final : public c10::OperatorKernel {
 * >   public:
 * >     Tensor operator()(Tensor a, Tensor b) {...}
 * >   };
 * > }
 *
 * The kernel class is allowed to have members but these are equivalent
 * to global variables. The kernel implementation is responsible for
 * preventing race conditions on them.
 *
 * See below for how to register this kernel with PyTorch.
 */
struct CAFFE2_API OperatorKernel {
  virtual ~OperatorKernel() = default;
};

namespace detail {
  // supported_primitive_arg_types defines which primitive types we allow in
  // kernel functions as arguments or returns.
  // Additionally, we support lists, dicts and optionals containing these types.
  using supported_primitive_arg_types = guts::typelist::typelist<
    int64_t,
    double,
    bool,
    std::string,
    at::Tensor,
    at::Scalar
  >;

  template<class T, bool AllowDeprecatedTypes, class Enable = void> struct assert_is_valid_input_type {
    assert_is_valid_input_type() {
    	std::cout << "In base " << std::endl;
      auto tmap = c10::getCustomClassTypeMap();
      TORCH_CHECK(c10::isCustomClassRegistered<T>(), "Tried to use undefined class as input argument");
    }
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes,
  	  std::enable_if_t<guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    // everything is ok, this is a primitive type
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<c10::optional<T>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
  };



  template<class Key, class Value, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<Dict<Key, Value>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<Value, AllowDeprecatedTypes> {
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value,
    		"You tried to register a kernel with an unsupported input type: Dict<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
  };

  template<class Key, class Value, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<std::unordered_map<Key, Value>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<Value, AllowDeprecatedTypes> {
    static_assert(AllowDeprecatedTypes,
    		"You tried to register a kernel with an unsupported input type: std::unordered_map<Key, Value>. Please use Dict<Key, Value> instead.");
    static_assert(guts::typelist::contains<impl::valid_dict_key_types, Key>::value,
    		"You tried to register a kernel with an unsupported input type: std::unordered_map<Key, Value> where Key is invalid. We only support int64_t, double, bool, and string.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<List<T>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value,
    		"You tried to register a kernel with an unsupported input type: List<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
  };

  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<std::vector<T>, AllowDeprecatedTypes>
  : assert_is_valid_input_type<T, AllowDeprecatedTypes> {
    static_assert(!std::is_same<T, at::Scalar>::value, "You tried to register a kernel with an unsupported input type: std::vector<Scalar>. Please use List<int64_t>, List<double> or Tensor instead.");
    // TODO static_assert(AllowDeprecatedTypes, "You tried to register a kernel with an unsupported input type: std::vector<T>. Please use List<T> instead.");
  };

  // The following specialisations of assert_is_valid_input_type are technically not
  // necessary since we would hit the base case and show an error message
  // there if they didn't exist, but we can show a better error message
  // in some common error scenarios.
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<float, T>::value>> {
    // There is no reason to support float when we have double. Keep the API lean.
    static_assert(guts::false_t<T>::value,
    		"You tried to register a kernel with an unsupported input type: float. Please use double instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<const char*, T>::value>> {
    static_assert(guts::false_t<T>::value,
    		"You tried to register a kernel with an unsupported input type: const char*. Please use std::string instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_same<std::vector<bool>, T>::value>> {
    static_assert(guts::false_t<T>::value,
    		"You tried to register a kernel with an unsupported input type: vector<bool>. Please use List<bool> instead.");
  };
  template<class T, bool AllowDeprecatedTypes>
  struct assert_is_valid_input_type<T, AllowDeprecatedTypes, std::enable_if_t<std::is_integral<T>::value && !guts::typelist::contains<supported_primitive_arg_types, T>::value>> {
    static_assert(guts::false_t<T>::value,
    		"You tried to register a kernel with an unsupported integral input type. Please use int64_t instead.");
  };

  template<class T, bool AllowDeprecatedTypes, class Enable = void> struct assert_is_valid_output_type {
    assert_is_valid_output_type() {
      auto tmap = getCustomClassTypeMap();
      TORCH_CHECK(c10::isCustomClassRegistered<T>(), "Tried to use undefined class as output");
    }
  };
}
}
