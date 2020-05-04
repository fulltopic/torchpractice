#include <torch/script.h>
#include <torch/torch.h>

//#include <Aten/Aten.h>
#include <random>
#include <memory>
#include <string>
#include <iostream>
#include <thread>


//#include <gtest/gtest.h>
//#include <ATen/core/boxing/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
//#include <torch/csrc/jit/script/function_schema_parser.h>
//#include <gmock/gmock.h>

#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/TensorTypeId.h>

using c10::RegisterOperators;
using c10::OperatorKernel;
using c10::TensorTypeId;
using c10::Stack;
using c10::guts::make_unique;
using c10::intrusive_ptr;
using c10::Dict;
using at::Tensor;
using std::unique_ptr;
using std::string;


using namespace std;
using namespace torch::nn;
using namespace torch::optim;

struct ErrorKernel final : public OperatorKernel {
  int64_t operator()(const Tensor&, int64_t) {
//    EXPECT_TRUE(false); // this kernel should never be called
	  std::cout << "Failure" << std::endl;
    return 0;
  }
};

struct IncrementKernel final : OperatorKernel {
  int64_t operator()(const Tensor& tensor, int64_t input) {
    return input + 1;
  }
};

struct IncrementKernel1 final : OperatorKernel {
  int64_t operator()(const Tensor& tensor, int64_t input) {
    return input * 2;
  }
};

template<class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
}

inline at::Tensor dummyTensor(c10::TensorTypeId dispatch_key) {
  auto* allocator = c10::GetCPUAllocator();
  int64_t nelements = 1;
  auto dtype = caffe2::TypeMeta::Make<float>();
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(nelements * dtype.itemsize()),
    allocator,
    /*resizable=*/true);
  return at::detail::make_tensor<c10::TensorImpl>(storage_impl, dispatch_key);
}

template<class... Args>
inline std::vector<c10::IValue> callOp(const c10::OperatorHandle& op, Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  c10::Dispatcher::singleton().callBoxed(op, &stack);
  return stack;
}

void expectCallsIncrement(TensorTypeId type_id) {
  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  if (!op.has_value()) {
	  std::cout << "no value of op " << std::endl;
  }else {
//	  std::cout << op.value() << std::endl;
  }
//  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(type_id), 5);

  std::cout << result.size() << ", " << result[0].toInt() << std::endl;
//  EXPECT_EQ(1, result.size());
//  EXPECT_EQ(6, result[0].toInt());
}

void test() {
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int",
		  RegisterOperators::options().kernel<ErrorKernel>(TensorTypeId::CPUTensorId));
  auto registrar2 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int",
		  RegisterOperators::options().kernel<ErrorKernel>(TensorTypeId::CUDATensorId));
  auto registrar3 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int",
		  RegisterOperators::options().kernel<IncrementKernel>(TensorTypeId::CPUTensorId));
  auto registrar4 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int",
		  RegisterOperators::options().kernel<ErrorKernel>(TensorTypeId::CUDATensorId));
  expectCallsIncrement(TensorTypeId::CPUTensorId);
}

void testThread() {
	unsigned int n = std::thread::hardware_concurrency();
	std::cout << "Hardware concurrency " << n << std::endl;

	char* omp = std::getenv("OMP_NUM_THREADS");
	std::cout << "omp num threads " << omp << std::endl;

	char* mkl = std::getenv("MKL_NUM_THREADS");
	std::cout << "mkl num threads " << mkl << std::endl;
}


int main() {
//	LSTM model(LSTMOptions(128, 64).layers(3).dropout(0.2).batch_first(0));
//	auto x = torch::randn({10, 16, 128}, torch::requires_grad());
//	auto output = model->forward(x);
//
//	cout << output.output.sizes() << endl;

//	test();
	testThread();
	return 0;
}
