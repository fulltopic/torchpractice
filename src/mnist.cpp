#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

const std::string kDataRoot = "./data";
const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize = 64;
const int64_t kNumberOfEpochs = 10;
const int64_t kLogInterval = 10;

struct Net: torch::nn::Module {
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Dropout2d conv2_drop;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;

	Net(): conv1(torch::nn::Conv2dOptions(1, 10, 5)),
			conv2(torch::nn::Conv2dOptions(10, 20, 5)),
			fc1(320, 50),
			fc2(50, 10){
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv2_drop", conv2_drop);
		register_module("fc1", fc1);
		register_module("fc2", fc2);
	}

	//No reference of input argument
	torch::Tensor forward(torch::Tensor x) {
		std::cout << "conv1, maxpool, relu " << std::endl;
		x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
		//TODO: The dropout
		std::cout << "conv2, maxpool, relu " << std::endl;
		x = torch::relu(torch::max_pool2d(conv2->forward(x), 2));
		x = x.view({-1, 320});
		std::cout << "fc1, relu " << std::endl;
		x = torch::relu(fc1->forward(x));
		std::cout << "dropout " << std::endl;
		x = torch::dropout(x, 0.5, is_training());
		std::cout << "fc2 " << std::endl;
		x = fc2->forward(x);

		std::cout << "Mnist output " << x.sizes() << std::endl;
		return torch::log_softmax(x, 1);
	}
};

template <typename DataLoader>
void train(size_t epoch,
		Net& model,
		torch::Device device,
		DataLoader& data_loader,
		torch::optim::Optimizer& optimizer,
		size_t dataset_size) {
	model.train();
	size_t batch_idx = 0;
	for (auto& batch: data_loader) {
		auto data = batch.data.to(device);
		auto targets = batch.target.to(device);

		std::cout << data.dim() << ", " << data.sizes() << std::endl;
		std::cout << "target " << targets.dim() << ", " << targets.sizes() << std::endl;

		optimizer.zero_grad();
		auto output = model.forward(data);
		auto loss = torch::nll_loss(output, targets);
		AT_ASSERT(!std::isnan(loss.template item<float>()));
		//TODO: loss knows how to backward
		loss.backward();
		optimizer.step();

		if(batch_idx ++ % kLogInterval == 0) {
		      std::printf(
		          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
		          epoch,
		          batch_idx * batch.data.size(0),
		          dataset_size,
		          loss.template item<float>());
		}
	}
}

template <typename DataLoader>
void test (Net& model,
		torch::Device device,
		DataLoader& data_loader,
		size_t dataset_size) {
	torch::NoGradGuard no_grad;
	model.eval();
	double test_loss = 0;
	int32_t correct = 0;
	for (const auto& batch: data_loader) {
		auto data = batch.data.to(device);
		auto targets = batch.target.to(device);
		auto output = model.forward(data);
		test_loss += torch::nll_loss(output,
									targets,
									{},
									at::Reduction::Sum)
									.template item<float>();
		auto pred = output.argmax(1);
		correct += pred.eq(targets).sum().template item<int64_t>();
	}

	test_loss /= dataset_size;
	std::printf(
	      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
	      test_loss,
	      static_cast<double>(correct) / dataset_size);
}

auto main() -> int {
	torch::manual_seed(1);

	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! training on GPU." << std::endl;
		device_type = torch::kCUDA;
	}else {
		std::cout << "Training on CPU" << std::endl;
		device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	Net model;
	model.to(device);

	auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
						.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
						.map(torch::data::transforms::Stack<>());
	const size_t train_dataset_size = train_dataset.size().value();
	auto train_loader =
			torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
					std::move(train_dataset), kTrainBatchSize);
	auto test_dataset = torch::data::datasets::MNIST(
						kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
						.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
						.map(torch::data::transforms::Stack<>());
	const size_t test_dataset_size = test_dataset.size().value();
	auto test_loader =
			torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

//	auto data = train_loader.get();

	torch::optim::SGD optimizer (
			model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

	for (size_t epoch = 1; epoch <= kNumberOfEpochs; epoch ++) {
		train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
		test(model, device, *test_loader, test_dataset_size);
	}

}
