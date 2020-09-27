/*
 * testtool.cpp
 *
 *  Created on: Sep 11, 2020
 *      Author: zf
 */

#include <string>
#include <iostream>
#include <memory>

#include <iosfwd>
#include <sstream>
#include <thread>
#include <condition_variable>
#include <mutex>

#include <time.h>

#include <torch/torch.h>

#include "rltest/l2rlovernet.h"
#include "rltest/l2rlcombnet.h"
#include "rltest/l2net.h"

#include "lmdbtools/LmdbReaderWrapper.h"
#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"
#include "lmdbtools/Lmdb2RowDataDefs.h"

#include "utils/dataqueue.hpp"

using std::cout;
using std::endl;
using std::vector;

using torch::Tensor;

using rltest::GRUL2Net;

struct TestModule: public torch::nn::Cloneable<TestModule> {
	torch::nn::GRU gru;
	torch::nn::Linear fc;
	int testMember;

	TestModule(int testValue):
		gru(torch::nn::GRUOptions(128, 1024).num_layers(2).batch_first(true)),
			fc(1024,16),
			testMember(testValue) {
		register_module("gru", gru);
		register_module("fc", fc);
	}

	TestModule(const TestModule& other) = default;
	TestModule& operator=(TestModule& other) = default;
	TestModule(TestModule&& other) = default;
	TestModule& operator=(TestModule&& other) = default;

	~TestModule() = default;

	void reset() override {
		register_module("gru", gru);
		register_module("fc", fc);
	}

	torch::Tensor forward(torch::Tensor input) {
		return input;
		//do sth.
	}
};

static std::string validDbPath = "/home/zf/workspaces/res/dbs/lmdb5rowscenetestvalid";
static LmdbSceneReader<LmdbDataDefs> validReader(validDbPath);

static void processActionInput(Tensor& input, Tensor& label) {
	auto inputPtr = input.accessor<float, 3>();
	auto labelPtr = label.data_ptr<long>();
	for (int j = 0; j < label.size(0); j ++) {
		if (labelPtr[j] >= 34) {
//				cout << "Label is -------------------------> " << labelPtr[j] << endl;
			for (int k = 34; k < 42; k ++) {
				inputPtr[j][1][k] = 1;
			}
		}
	}
}

static void evalAccu(Tensor output, Tensor labelTensor) {
	Tensor indices = torch::argmax(output, -1);
	Tensor diff = torch::sub(labelTensor, indices);

	int total = labelTensor.size(0);
	auto matched = total - diff.nonzero().size(0);
	float accu = (float)matched / total;

	cout << "accu: " << accu << endl;
	//TODO: print out result
}

template <typename NetType>
static Tensor evalStepModel(std::shared_ptr<NetType> net, LmdbSceneReader<LmdbDataDefs>& validReader) {
	validReader.reset();
	vector<Tensor> validInputs;
	vector<Tensor> validLabels;
	std::tie(validInputs, validLabels) = validReader.next(16);

	Tensor input = validInputs[0];
	Tensor label = validLabels[0];
	processActionInput(input, label);
	//TODO: Where to put div(4)
	input = input.div(4);

	net->eval();
	const int seqLen = input.size(0);
	cout << "input sizes " << input.sizes() << endl;
	cout << "seqLen: " << seqLen << endl;

	vector<Tensor> outputs;
	outputs.reserve(seqLen);
	Tensor hState = torch::zeros({2, 1, 2048});
	for (int i = 0; i < seqLen; i ++) {
		Tensor netInput = input[i].view({1, 1, 360});
		auto output = net->forward({netInput, hState, torch::tensor(i)});
		hState = output[1];
		cout << "step output sizes: " << output[0].sizes() << endl;
		outputs.push_back(output[0].view({1, 42}));
	}

	cout << "output sizes " << outputs[0].sizes() << endl;
	Tensor outTensor = torch::cat(outputs, 0);
//	cout << "outputTensor " << outTensor << endl;
	cout << "label " << label.sizes() << endl;

	Tensor labelTensor = torch::cat(label, 0);
	auto loss = torch::nll_loss(outTensor, labelTensor);

	evalAccu(outTensor, labelTensor);

	return loss;
}

template<typename NetType, typename OptType>
static void stepNet(std::shared_ptr<NetType>& net, OptType& optimizer) {
	Tensor loss = evalStepModel(net, validReader);
	optimizer.zero_grad();
	loss.backward();
	optimizer.step();
}

static bool compTensor(Tensor& v0, Tensor& v1) {
	if (v1.is_same(v0)) {
		cout << "Not clone, just imp pointer copied" << endl;
		return false;
	}

	if (v0.dim() != v1.dim()) {
		cout << " have different dim " << v0.dim() << " != " << v1.dim() << endl;
		return false;
	}
	auto numel0 = v0.numel();
	auto numel1 = v1.numel();
	if (numel0 != numel1) {
		cout << "Different cell number: " << numel0 << " != " << numel1 << endl;
		return false;
	}

	auto size0 = v0.sizes();
	auto size1 = v1.sizes();
	for (int i = 0; i < v0.dim(); i ++) {
		if (v0.size(i) != v1.size(i)) {
			cout << "Size does match at dim " << i << " " << v0.size(i) << " != " << v1.size(i) << endl;
			return false;
		}
	}

	auto data0 = v0.data_ptr<float>();
	auto data1 = v1.data_ptr<float>();
	for (int i = 0; i < numel0; i ++) {
		if (data0[i] != data1[i]) {
			cout << "Different value of element " << i << ": " << data0[i] << " != " << data1[i] << endl;
			return false;
		}
	}

	return  true;
}



template <class NetType>
static bool compNet(std::shared_ptr<NetType> net0, std::shared_ptr<NetType> net1) {
	cout << endl << endl;
	cout << "Compare nets " << endl;

	auto params0 = net0->named_parameters(true);
	auto params1 = net1->named_parameters(true);
	for (auto ite = params0.begin(); ite != params0.end(); ite ++) {
		auto key = ite->key();
		cout << "Test param " << key << ": ---------------------------> " << endl;

		Tensor v0 = ite->value();
		Tensor* v1 = params1.find(key);
		if (v1->is_same(v0)) {
			cout << "Not clone, just imp pointer copied" << endl;
			return false;
		}

		if (v1 == nullptr) {
			cout << "Could not find " << key << " in net1 " << endl;
			return false;
		}

		if (v0.dim() != v1->dim()) {
			cout << "Param " << key << " have different dim " << v0.dim() << " != " << v1->dim() << endl;
			return false;
		}
		auto numel0 = v0.numel();
		auto numel1 = v1->numel();
		if (numel0 != numel1) {
			cout << "Different cell number: " << numel0 << " != " << numel1 << endl;
			return false;
		}

		auto size0 = v0.sizes();
		auto size1 = v1->sizes();
		for (int i = 0; i < v0.dim(); i ++) {
			if (v0.size(i) != v1->size(i)) {
				cout << "Size does match at dim " << i << " " << v0.size(i) << " != " << v1->size(i) << endl;
				return false;
			}
		}

		auto data0 = v0.data_ptr<float>();
		auto data1 = v1->data_ptr<float>();
		for (int i = 0; i < numel0; i ++) {
			if (data0[i] != data1[i]) {
				cout << "Different value of element " << i << ": " << data0[i] << " != " << data1[i] << endl;
				return false;
			}
		}
	}

	vector<Tensor> buffers0 = net0->buffers(true);
	vector<Tensor> buffers1 = net1->buffers(true);
	cout << "Buffer size: " << buffers0.size() << endl;
	for (int i = 0; i < buffers0.size(); i ++) {
		if (!buffers0[i].equal(buffers1[i])) {
			cout << "Buffer at " << i << " does not match " << endl;
			return false;
		}
	}

	auto children0 = net0->children();
	auto children1 = net1->children();
	if (children0.size() != children1.size()) {
		cout << "Different size of children: " << children0.size() << " != " << children1.size() << endl;
		return false;
	}
	for (int i = 0; i < children0.size(); i ++) {
		cout << "Name" << i << ": " << children0[i]->name() << ", " << children1[i]->name() << endl;
	}


	return true;
}


static void testCloneable() {
	std::shared_ptr<TestModule> net(new TestModule(27));
	torch::optim::RMSprop optimizer(net->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99));
	//forward and backward of net


	auto copy = net->clone();
	std::shared_ptr<TestModule> cpyNet = std::dynamic_pointer_cast<TestModule>(copy);
	torch::optim::RMSprop cpyOptimizer(cpyNet->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99));
	//forward and backward of cpyNet
}

static void testClone() {
	const std::string modelPath = "/home/zf/workspaces/workspace_cpp/aws/GRU2L2048MaskNet_140000_0.002000_1593719779.pt";

	std::shared_ptr<rltest::GRUL2Net> net(new rltest::GRUL2Net(27));
	net->initParams();
	net->loadModel(modelPath);

	torch::optim::RMSprop optimizer(net->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99));
	auto copy = net->clone();
	std::shared_ptr<rltest::GRUL2Net> cpyNet = std::dynamic_pointer_cast<rltest::GRUL2Net>(copy);

	bool isSame = compNet(net, cpyNet);
	cout << "Net is same: " << isSame << endl;


	Tensor loss = evalStepModel(net, validReader);
}


static void testLoadOverall() {
	std::shared_ptr<rltest::GRUL2OverNet> net(new rltest::GRUL2OverNet(27));
	const std::string modelPath = "/home/zf/workspaces/workspace_cpp/aws/GRU2L2048MaskNet_140000_0.002000_1593719779.pt";
	net->loadL2Model(modelPath);

	torch::optim::RMSprop optimizer(net->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99));



	auto copy = net->clone();
	std::shared_ptr<rltest::GRUL2OverNet> cpyNet = std::dynamic_pointer_cast<rltest::GRUL2OverNet>(copy);
	bool isSame = compNet(net, cpyNet);
	cout << "Is same immediately after clone: " << isSame << endl << endl;


//	std::string validDbPath = "/home/zf/workspaces/res/dbs/lmdb5rowscenetestvalid";
//	cout << "validDbPath: " << validDbPath << endl;
//	LmdbSceneReader<LmdbDataDefs> validReader(validDbPath);

//	Tensor loss = evalStepModel(net, validReader);
//	optimizer.zero_grad();
//	loss.backward();
//	optimizer.step();
	stepNet(net, optimizer);

	isSame = compNet(net, cpyNet);
	cout << "Is same after backward update " << isSame << endl << endl;


	copy = net->clone();
	cpyNet = std::dynamic_pointer_cast<rltest::GRUL2OverNet>(copy);
	isSame = compNet(net, cpyNet);
	cout << "Is same after clone again " << isSame << endl;
}

static void testOpt() {
	std::shared_ptr<rltest::GRUL2OverNet> net(new rltest::GRUL2OverNet(27));
	torch::optim::RMSprop optimizer(net->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99));
	auto& groups = optimizer.param_groups();
//	vector<Tensor>& plainParams = optimizer.parameters();
//
//	cout << "plain paramters size: " << plainParams.size() << endl;
//	cout << "group size: " << groups.size() << endl;
//
//	vector<Tensor>& params = groups[0].params();
//	cout << "param size: " << params.size() << endl;
//
//	params.clear();
//	cout << "param size after clear: " << params.size() << endl;
//	cout << "plainParams size: " << plainParams.size() << endl;
//

//	auto newParams = optimizer.param_groups()[0].params();
//	auto newPlainParams = optimizer.parameters();
//	cout << "param size after refretch: " << newParams.size() << endl;
//	cout << "plainParams size after refretch: " << newPlainParams.size() << endl;

	groups.clear();

	auto newGroups = optimizer.param_groups();
//	auto newPlainParams = optimizer.parameters();
	cout << "group size after refretch: " << newGroups.size() << endl;
//	cout << "plainParams size after refretch: " << newPlainParams.size() << endl;
//	optimizer.add_param_group(net->parameters());
//	cout << "param size after add: " <<

	optimizer.add_param_group(net->parameters());
	cout << "group size after reset: " << optimizer.param_groups().size() << endl;
	cout << "param in group after reset: " << optimizer.param_groups()[0].params().size() << endl;
	cout << "plainParams size after reset: " << optimizer.parameters().size() << endl;
}

static void compOptParams(torch::optim::RMSprop& optimizer, torch::optim::RMSprop& cpyOptimizer) {
	auto params = optimizer.parameters();
	auto cpyParams = cpyOptimizer.parameters();
	if (params.size() != cpyParams.size()) {
		cout << "Orig params have diff size: " << params.size() << " != " << cpyParams.size() << endl;
	} else {
		for (int i = 0; i < params.size(); i ++) {
			compTensor(params[i], cpyParams[i]);
		}
	}
}

static void testOptCpy() {
	std::shared_ptr<rltest::GRUL2OverNet> net(new rltest::GRUL2OverNet(27));
	torch::optim::RMSprop optimizer(net->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99));
//	auto& groups = optimizer.param_groups();

	auto copy = net->clone();
	std::shared_ptr<rltest::GRUL2OverNet> cpyNet = std::dynamic_pointer_cast<rltest::GRUL2OverNet>(copy);
	torch::optim::RMSprop cpyOptimizer(cpyNet->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99));

	cout << "comp params before update" << endl;
	compOptParams(optimizer, cpyOptimizer);

	stepNet(net, optimizer);

//	Tensor loss = evalStepModel(net, validReader);
//	optimizer.zero_grad();
//	loss.backward();
//	optimizer.step();

	cout << "comp params after update" << endl;
	compOptParams(optimizer, cpyOptimizer);

}

static void parseOpt(torch::optim::RMSprop& optimizer, std::shared_ptr<rltest::GRUL2OverNet> net) {
//	std::shared_ptr<rltest::GRUL2OverNet> net(new rltest::GRUL2OverNet(27));
//	torch::optim::RMSprop optimizer(net->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99).momentum(0.1));
//
//	stepNet(net, optimizer);

	auto& states = optimizer.state();
	auto& params = optimizer.parameters();

//	c10::guts::to_string(p.unsafeGetTensorImpl()
	auto namesParams = net->named_parameters(true);
	for (auto ite = namesParams.begin(); ite != namesParams.end(); ite ++) {
		auto key = ite->key();
		cout << "Test param " << key << ": ---------------------------> " << endl;

		Tensor value = ite->value();
		auto valueStr = c10::guts::to_string(value.unsafeGetTensorImpl());
		cout << "The key to state dict is: " << valueStr << endl;

		if (states.find(valueStr) == states.end()) {
			cout << "tensor not defined " << endl;
		} else {
			cout << "state defined" << endl;
			const torch::optim::RMSpropParamState& curr_state_ = static_cast<const torch::optim::RMSpropParamState&>(*(states.at(valueStr).get()));
			if(curr_state_.momentum_buffer().defined()) {
				cout << "curr_state momentum buffer defined " << endl;
			}
			if(curr_state_.grad_avg().defined()) {
				cout << "curr_state grad_avg defined " << endl;
			}
		}
	}

	//TODO: Test set the states tensors directly
}

static void cloneOpt(torch::optim::RMSprop& opt0, torch::optim::RMSprop& opt1,
		std::shared_ptr<rltest::GRUL2OverNet>& net0, std::shared_ptr<rltest::GRUL2OverNet>& net1) {
	auto namedParams0 = net0->named_parameters(true);
	auto namedParams1 = net1->named_parameters(true);
	auto& states0 = opt0.state();
	auto& states1 = opt1.state();
	states1.clear();

	for (auto ite = namedParams0.begin(); ite != namedParams0.end(); ite ++) {
		auto key = ite->key();
		cout << "To deal with param " << key << endl;

		Tensor value0 = ite->value();
		auto value0Str = c10::guts::to_string(value0.unsafeGetTensorImpl());

		if (states0.find(value0Str) == states0.end()) {
			cout << "No state defined for param " << key << endl;
			continue;
		}

//		const torch::optim::RMSpropParamState& currState
//			= static_cast<const torch::optim::RMSpropParamState&>(*(states0.at(value0Str).get()));
//		std::unique_ptr<torch::optim::RMSpropParamState>& currState = states0.at(value0Str);
		auto& currState = states0.at(value0Str);
//		cout << "currState type " << typeid(currState).name() << endl;
		auto stateClone = currState->clone();
//		cout << "clone state type " << typeid(stateClone).name() << endl;
//		auto stateImpl = currState.release();
//		cout << "stateImpl type " <<  typeid(stateImpl).name() << endl;
//		cout << "curstate type " << typeid(states0.at(value0Str)).name() << endl;
//		cout << "clone type " << typeid(currState->clone()).name() << endl;
		Tensor value1 = namedParams1[key];
		auto value1Str = c10::guts::to_string(value1.unsafeGetTensorImpl());
		states1[value1Str] = std::move(stateClone);
//		cout << "Set state for param " << value1Str << endl;
	}
}

static void testCpyOpt() {
	//net0 run
	std::shared_ptr<rltest::GRUL2OverNet> net0(new rltest::GRUL2OverNet(27));
	torch::optim::RMSprop opt0(net0->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99).momentum(0.1));

	stepNet(net0, opt0);
	cout << "------------------------------> Net0 created and steped ";

	//net1 clone
	auto copy = net0->clone();
	std::shared_ptr<rltest::GRUL2OverNet> net1 = std::dynamic_pointer_cast<rltest::GRUL2OverNet>(copy);
	auto& options = opt0.defaults();
//	auto test = std::make_shared<decltype(options)>(decltype(options)(options));
//	cout << "test type: " << typeid(test).name() << endl;
	cout << "Options type: " << typeid(options).name() << endl;
	cout << "Deltype " << typeid(decltype(options)).name() << endl;
	torch::optim::RMSprop opt1(net1->parameters(), dynamic_cast<torch::optim::RMSpropOptions&>(options));
//	torch::optim::RMSprop opt1(net1->parameters(), static_cast<typeid(options)&>(options));

	std::ostringstream buf;
	torch::serialize::OutputArchive output_archive;
	opt0.save(output_archive);
	output_archive.save_to(buf);

	std::istringstream inStream(buf.str());
	torch::serialize::InputArchive input_archive;
	input_archive.load_from(inStream);
	opt1.load(input_archive);
	parseOpt(opt1, net1);

	//net1 step
	stepNet(net1, opt1);

	//net0 clone
	auto copy1 = net1->clone();
	net0 = std::dynamic_pointer_cast<rltest::GRUL2OverNet>(copy1);
	auto& options1 = opt1.defaults();
	torch::optim::RMSprop opt2(net0->parameters(), static_cast<torch::optim::RMSpropOptions&>(options1));

	std::ostringstream buf1;
	torch::serialize::OutputArchive output_archive1;
	opt1.save(output_archive1);
	output_archive.save_to(buf1);

	std::istringstream inStream1(buf1.str());
	torch::serialize::InputArchive input_archive1;
	input_archive1.load_from(inStream);
	opt2.load(input_archive1);
	parseOpt(opt2, net0);
//	stepNet(net1, opt1);

}

static void testCloneTurn() {
	std::shared_ptr<rltest::GRUL2OverNet> net0(new rltest::GRUL2OverNet(27));
	torch::optim::RMSprop opt0(net0->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99).momentum(0.1));

	stepNet(net0, opt0);
	cout << "------------------------------> Net0 created and steped ";

	auto copy = net0->clone();
	std::shared_ptr<rltest::GRUL2OverNet> net1 = std::dynamic_pointer_cast<rltest::GRUL2OverNet>(copy);

	auto& options = opt0.defaults();

//	torch::optim::RMSprop opt1(net1->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99));
	torch::optim::RMSprop opt1(net1->parameters(), static_cast<torch::optim::RMSpropOptions&>(options));
//	torch::optim::RMSprop opt1(net1->parameters(), options));
	cloneOpt(opt0, opt1, net0, net1);
	cout << "------------------------------> Net0 cloned into net1 and opt0 copied into opt1" << endl;
	stepNet(net1, opt1);

	copy = net1->clone();
	net0 = std::dynamic_pointer_cast<rltest::GRUL2OverNet>(copy);
	cout << "------------------------------> Net1 cloned into net0" << endl;
	torch::optim::RMSprop opt2(net0->parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99).momentum(0.1));
	cout << "------------------------------> opt2 created for net0 " << endl;
	cloneOpt(opt1, opt2, net1, net0);
	cout << "------------------------------> opt2 copied from opt1" << endl;
}




static void testUint() {
	uint32_t lastSeq = (0 - 1);
	uint32_t curSeq = 0;

	cout << "last, cur = " << (uint32_t)lastSeq << ", " << (uint32_t)curSeq << endl;
	bool isSame = (lastSeq == (curSeq - 1));
	cout << "isSame: " << isSame << endl;

	uint32_t lastIndex = (curSeq - 1) % 2;
	cout << "lastIndex " << lastIndex << endl;


}

static void testDataQueue() {
	R1WmQueue<int, 16> q;

	for (int i = 0; i < 16; i ++) {
		if (!q.push(std::move(i))) {
			cout << "Failed to push " << i << endl;
			break;
		}
	}
	cout << "More push: " << q.push(16) << endl;
	cout << "size: " << q.size() << endl;

	cout << "popped: ";
	while (!q.isEmpty()) {
		int d = q.pop();
		cout << d << ", ";
	}
	cout << endl;
}

const int threadNum = 4;
const int writeNum = 16;
const struct timespec layback { 0, 500 };
std::mutex lock;
std::condition_variable cond;
volatile bool started = false;

vector<vector<int>> writtenData(threadNum, vector<int>(16, -10));

static void readTask(R1WmQueue<int, 16>& q) {
	{
		std::unique_lock<std::mutex> guard(lock);
		started = true;
		cond.notify_all();
	}

//	vector<int>& data = readData[index];
	int readNum = 0;
	while (readNum < writeNum * threadNum) {
		while (q.isEmpty()) {
			nanosleep(&layback, nullptr);
//			sleep(1);
		}
		int data = q.pop();
		cout << "read " << data << endl;
		readNum ++;
	}
}

static void writeTask(R1WmQueue<int, 16>& q) {
	{
		if (!started) {
			std::unique_lock<std::mutex> guard(lock);
			cond.wait(guard);
		}
	}

	for (int i = 0; i < writeNum; i ++) {
		while (!q.push(std::move(i))) {
			nanosleep(&layback, nullptr);
//			sleep(1);
		}
	}
}

static void testDataQThread() {
	R1WmQueue<int, 16> q;
	vector<std::thread> ts;

	for (int i = 0; i < threadNum; i ++) {
		ts.push_back(std::move(std::thread(writeTask, std::ref(q))));
	}
	readTask(q);

	for (int i = 0; i < threadNum; i ++) {
		ts[i].join();
	}
}

static void testVector() {
	vector<vector<int>> datas(6);
	datas[0] = vector<int>(2, 0);
	datas[1] = vector<int>(3, 1);

	for (auto data: datas[1]) {
		cout << "data: " << data << endl;
	}
}

struct TestRefData {
	int data;

	TestRefData(int i): data(i) {
		cout << "TestRefData constructor " << endl;
	}

	TestRefData(const TestRefData& other): data(other.data) {
		cout << "TestRefData copy constructor" << endl;
	}
};

struct TestRefWrapper {
	TestRefData& data;

	TestRefWrapper(TestRefData& input): data(input) {}

	void add() {
		data.data = data.data + 1;
	}

	void update() {
		data = TestRefData(data.data + 10);
	}

	void printData() {
		cout << "data = " << data.data << endl;
	}
};

static void testRefFunc () {
//	TestRefData data1(10);
//	TestRefData data2(20);
//
//	TestRefWrapper ref1(data1);
//	TestRefWrapper ref2(data2);
//
//	ref1.add();
//	ref1.printData();
//	ref1.update();
//	ref1.printData();
//
//	ref2.printData();

//	std::vector<TestRefData> datas(2, TestRefData(20));

	std::vector<TestRefData> datas;
	for (int i = 0; i < 3; i ++) {
		datas.push_back(20);
	}
}


static void testZeros () {
	Tensor t0 = torch::zeros({2, 2});
	Tensor t1 = torch::zeros({2, 2});

	cout << "t0 " << (void*)(t0.unsafeGetTensorImpl()) << endl;
	cout << "t1 " << (void*)(t1.unsafeGetTensorImpl()) << endl;
}

static void testPad () {
	const int realSize = 28;
	const int expSize = 27;
	Tensor t = torch::zeros({2, realSize});
	float* tData = t.data_ptr<float>();
	for (int i = 0; i < t.numel(); i ++) {
		tData[i] = i;
	}
	cout << "Before pad " << endl;
	cout << t << endl;

	t = torch::constant_pad_nd(t, {0, (expSize - realSize)});
	cout << "After pad" << endl;
	cout << t << endl;
}
int main(int argc, char** argv) {
//	testClone();
//	testCloneable();
//	testLoadOverall();

//	testOpt();
//	testUint();

//	parseOpt();
//	testCloneTurn();

//	testCpyOpt();

//	testDataQueue();
//	testDataQThread();

//	testVector();
//	testRefFunc();

	testPad();

//	testZeros();
}
