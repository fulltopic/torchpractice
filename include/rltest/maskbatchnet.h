#include <torch/torch.h>

#include <string.h>
#include <string>
#include <vector>

namespace rltest {
struct GRUMaskNet: torch::nn::Module {
private:
//	const unsigned int numInputs;
//	const unsigned int numActOutput;
	torch::nn::GRU gru0;
	torch::nn::BatchNorm1d batchNorm0;
	torch::nn::Linear fc;

	torch::nn::Linear valueFc;

	const int seqLen;

	std::vector<torch::nn::BatchNorm1d> stepBatchNorms;

public:
	GRUMaskNet(int inSeqLen);
	GRUMaskNet(GRUMaskNet& other) = delete;
	GRUMaskNet& operator=(GRUMaskNet& other) = delete;
	GRUMaskNet(GRUMaskNet&& other) = delete;
	GRUMaskNet& operator=(GRUMaskNet&& other) = delete;

	~GRUMaskNet() = default;


	void initParams();
	void loadModel(const std::string modelPath);

	torch::Tensor inputPreprocess(torch::Tensor input);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs, bool isTrain);
	std::vector<torch::Tensor> forward (std::vector<torch::Tensor> inputs);
	torch::Tensor createHState();
	void reset();

	torch::Tensor getLoss(std::vector<torch::Tensor> inputTensors);

};
}
