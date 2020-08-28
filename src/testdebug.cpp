#include <torch/torch.h>
#include <iostream>

using namespace torch;

int main() {
	std::cout << "Create a tensor" << std::endl;
	Tensor t0 = torch::rand({2, 2}, TensorOptions().requires_grad(true));
	Tensor t1 = torch::rand({2, 2}, TensorOptions().requires_grad(true));


	Tensor a = torch::mm(t0, t1);
//	std::cout << "y add" << std::endl;
	Tensor b = a + t1;
	std::cout << "x add" << std::endl;
	Tensor c = b + t0;
	Tensor d = torch::sin(c);
	Tensor e = d.mean();
//	std::cout << std::endl << "----------------------------> testdebug::backward" << std::endl;
//	Tensor t = z.detach();
	e.backward();
//	t.backward();

//	std::cout << tensor0.grad() << std::endl;
//
//
//	std::cout << std::endl << std::endl << std::endl;
//
//	y = tensor0 + tensor1;
//	z = y.sin();
//	z = z.sum();
//	z.backward();
//	std::cout << "------------------------> End of second " << std::endl;

//	Tensor z = y.detach();
//	z.backward();

//	y.backward();
}
