/*
 * LmdbReader.h
 *
 *  Created on: Jan 1, 2020
 *      Author: zf
 */

#ifndef SRC_LMDBTOOLS_LMDBREADER_H_
#define SRC_LMDBTOOLS_LMDBREADER_H_
#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/proto/caffe2_pb.h>
#include <caffe2/proto/caffe2_legacy.pb.h>
#include <caffe2/core/logging.h>
#include <caffe2/core/blob_serialization.h>
#include <torch/torch.h>

#include <string>

template<typename DataDefs>
class LmdbSceneReader {
public:
	const std::string DbType = "lmdb";
//	const static int LabelLen = 42;

	explicit LmdbSceneReader(const std::string path);
	LmdbSceneReader(const LmdbSceneReader& other) = delete;
	LmdbSceneReader& operator= (const LmdbSceneReader& other) = delete;

	std::pair<torch::Tensor, torch::Tensor> next();
//	std::pair<torch::TensorList, torch::TensorList> next(const int batchSize);
	std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> next(const int batchSize);
	std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> next(const int batchSize, const int seqLen);

	bool hasNext();
	void reset();
	virtual ~LmdbSceneReader();

private:
	const std::string dbPath;
	//TODO: Make it const
//	DataDefs dataDefs;

	std::unique_ptr<caffe2::db::DB> db;
	std::unique_ptr<caffe2::db::Cursor> cursor;


	std::vector<int64_t> dataDim;
	std::vector<int64_t> labelDim;


//	void setDims();
};

template<typename DataDefs>
LmdbSceneReader<DataDefs>::LmdbSceneReader(const std::string path): dbPath(path),
	db(caffe2::db::CreateDB(DbType, dbPath, caffe2::db::READ))
{
	cursor = db->NewCursor();
	dataDim = DataDefs::GetDataDim();
	labelDim = DataDefs::GetLabelDim();
	std::cout << "Reader db of path: " << dbPath << std::endl;
	for (int i = 0; i < dataDim.size(); i ++) {
		std::cout << dataDim[i] << ", ";
	}
	std::cout << std::endl;
}

template<typename DataDefs>
LmdbSceneReader<DataDefs>::~LmdbSceneReader() {
	db->Close();
}

template<typename DataDefs>
inline bool LmdbSceneReader<DataDefs>::hasNext() {
	return cursor->Valid();
}

template<typename DataDefs>
void LmdbSceneReader<DataDefs>::reset() {
	cursor->SeekToFirst();
}

template<typename DataDefs>
std::pair<torch::Tensor, torch::Tensor> LmdbSceneReader<DataDefs>::next() {
	const auto key = cursor->key();
//	std::cout << "Read key " << key << std::endl;

	caffe2::TensorProtos protos;
	protos.ParseFromString(cursor->value());
//	std::cout << protos.protos_size() << std::endl;

	std::vector<torch::Tensor> datas;
	std::vector<torch::Tensor> labels;
	torch::TensorOptions options;
	options = options.device(c10::DeviceType::CPU);
//	std::cout << "Options: " << options.device() << std::endl;

	for (int protoIndex = 0; protoIndex < protos.protos_size(); protoIndex ++) {
		auto proto = protos.protos(protoIndex);
		auto data = proto.int32_data();

		//TODO: Would the vector be copied
		std::vector<int> dataVec(data.data(), data.data() + data.size());


		if (protoIndex % 2 == 0) {
//			std::cout << "Orig size " << data.size() << std::endl;
//			std::cout << "data dim " << dataDim << std::endl;
			std::vector<float> newDataVec(dataVec.begin(), dataVec.end());
			auto dataTensor =  torch::tensor(newDataVec, options).view(dataDim).requires_grad_(true);
//			std::cout << "dataTensor " << dataTensor.device() << std::endl;
//			auto tensor = torch::tensor(newDataVec, options).view(dataDim);
//			auto tensor = dataTensor.view(dataDim);
//			tensor.to(torch::kCPU);
//			std::cout << "Device " << tensor.device() << std::endl;
//			std::cout << tensor << std::endl;
			datas.push_back(dataTensor);
//			for (int i = 0; i < proto.dims_size(); i ++) {
//				dims.push_back(proto.dim)
//			}
//			std::cout << "Read data tensor " << std::endl;
		} else {
//			std::cout << "Orig size " << data.size() << std::endl;
//			std::cout << "data dim " << dataDim << std::endl;
			auto tensor = torch::tensor(dataVec).to(torch::kCPU).view(labelDim);
//			auto tensor = torch::zeros({1, 42});
//			auto ptr = static_cast<float*>(tensor.data_ptr());
//			ptr[dataVec[0]] = 1;
			labels.push_back(tensor);
//			std::cout << "Read label tensor " << std::endl;
		}
	}

	cursor->Next();

	auto data = torch::cat(datas, 0).toType(c10::ScalarType::Float).to(torch::kCPU);
	auto label = torch::cat(labels, 0).toType(c10::ScalarType::Long).to(torch::kCPU);


	return std::make_pair(std::move(data), std::move(label));
}

template<typename DataDefs>
//std::pair<torch::TensorList, torch::TensorList> LmdbSceneReader<DataDefs>::next(const int batchSize) {
std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> LmdbSceneReader<DataDefs>::next(const int batchSize) {
	std::vector<torch::Tensor> dataVec;
	std::vector<torch::Tensor> labelVec;

	for (int i = 0; i < batchSize; i ++) {
		if (!hasNext()) {
			break;
		}

		torch::Tensor dataTensor;
		torch::Tensor labelTensor;

		std::tie(dataTensor, labelTensor) = next();
		dataVec.push_back(dataTensor);
		labelVec.push_back(labelTensor);

//		std::cout << "dataTensor " << dataTensor.sizes() << std::endl;
	}

	return std::make_pair(std::move(dataVec), std::move(labelVec));
//	torch::TensorList dataList(dataVec);
//	torch::TensorList labelList(labelVec);
//
//	std::cout << "dataList " << dataList.size() << std::endl;
//	std::cout << dataList[0].sizes() << std::endl;
//	std::cout << dataList.data() << std::endl;
//	std::cout << dataList[0].device() << std::endl;
////	return std::make_pair(dataVec, labelVec);
//	return std::make_pair(dataList, labelList);
}

template<typename DataDefs>
//std::pair<torch::TensorList, torch::TensorList> LmdbSceneReader<DataDefs>::next(const int batchSize) {
std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
	LmdbSceneReader<DataDefs>::next(const int batchSize, const int seqLen) {
	std::vector<torch::Tensor> dataVec;
	std::vector<torch::Tensor> labelVec;

	int count = 0;
	while (count < batchSize && hasNext()) {
		torch::Tensor dataTensor;
		torch::Tensor labelTensor;

		std::tie(dataTensor, labelTensor) = next();
		if (dataTensor.size(0) >= seqLen) {
			dataVec.push_back(dataTensor.narrow(0, dataTensor.size(0) - seqLen, seqLen));
			labelVec.push_back(labelTensor.narrow(0, dataTensor.size(0) - seqLen, seqLen));
			count ++;
		}
//		std::cout << "dataTensor " << dataTensor.sizes() << std::endl;
	}


	return std::make_pair(std::move(dataVec), std::move(labelVec));
//	torch::TensorList dataList(dataVec);
//	torch::TensorList labelList(labelVec);
//
//	std::cout << "dataList " << dataList.size() << std::endl;
//	std::cout << dataList[0].sizes() << std::endl;
//	std::cout << dataList.data() << std::endl;
//	std::cout << dataList[0].device() << std::endl;
////	return std::make_pair(dataVec, labelVec);
//	return std::make_pair(dataList, labelList);
}

#endif /* SRC_LMDBTOOLS_LMDBREADER_H_ */
