#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/proto/caffe2_pb.h>
#include <caffe2/proto/caffe2_legacy.pb.h>
#include <caffe2/core/logging.h>
#include <caffe2/core/blob_serialization.h>
#include <torch/torch.h>

#include "lmdbtools/LmdbReader.h"
#include "lmdbtools/LmdbDataDefs.h"

#include <matplotlibcpp.h>


#include <string>

using caffe2::db::Cursor;
using caffe2::db::DB;
using caffe2::db::Transaction;
using caffe2::CaffeDatum;
using caffe2::TensorProto;
using caffe2::TensorProtos;

using std::cout;
using std::endl;

const std::string inputDbPath = "/home/zf/workspaces/res/dbs/lmdb5rowscenetest";
const std::string outputDbPath = "/home/zf/workspaces/res/dbs/lmdb5rowscnntest";
const std::string dbType = "lmdb";


void createDB() {
	auto db = caffe2::db::CreateDB(dbType, outputDbPath, caffe2::db::NEW);
	auto transaction = db->NewTransaction();

	const std::string key = "4";

	caffe2::TensorProtos protos;
	auto dataProto = protos.add_protos();
	auto labelProto = protos.add_protos();

	dataProto->set_data_type(caffe2::TensorProto::INT32);
	dataProto->add_dims(1);
	dataProto->add_dims(16);

	labelProto->set_data_type(caffe2::TensorProto::INT32);
	labelProto->add_dims(1);
	labelProto->add_int32_data(3);

	cout << "All protos created " << endl;

	int data[4];
	for (int i = 0; i < 4; i ++) {
		data[i] = i;
		dataProto->add_int32_data(i);
	}
	cout << "Data prepared " << endl;

//	dataProto->set_byte_data(data, 4 * sizeof(int));
//	cout << "data injected " << endl;

	std::string value;
	protos.SerializeToString(&value);
	cout << "Bytes created" << endl;

	transaction->Put(key, value);
	transaction->Commit();
	cout << "trans committed" << endl;

//	db->Close();
}

static void testWrite() {
//	LmdbSceneReader<LmdbDataDefs> reader(inputDbPath);
//	torch::Tensor datas;
//	torch::Tensor labels;
//	std::tie(datas, labels) = reader.next();
//
//	std::cout << "Sizes " << datas[0].sizes() << std::endl;
//	std::cout << " " << labels.sizes() << std::endl;
//
//	const int dataDim1 = datas.size(1);
//	const int dataDim2 = datas.size(2);
//	const int labelDim = 1;

//	torch::TensorList dataList = datas.chunk(datas.size(0), 0);
//	std::cout << "Sizes " << dataList[0].sizes() << std::endl;
//	std::cout << "Next " << std::endl;
//	std::cout << "Sizes " << dataList.at(1).sizes() << std::endl;

	const int dataDim1 = 5;
	const int dataDim2 = 72;
	const int labelDim = 1;

	std::unique_ptr<caffe2::db::DB> readDb = caffe2::db::CreateDB(dbType, inputDbPath, caffe2::db::READ);
	auto cursor = readDb->NewCursor();
	const auto inputKey = cursor->key();
	std::cout << "Read key " << inputKey << std::endl;

	caffe2::TensorProtos inputProtos;
	inputProtos.ParseFromString(cursor->value());
	auto inputDataProto = inputProtos.protos(0);
	auto inputLabelProto = inputProtos.protos(1);
	auto inputData = inputDataProto.int32_data();
	auto inputLabel = inputLabelProto.int32_data();


	int seq = 33;
	auto db = caffe2::db::CreateDB(dbType, outputDbPath, caffe2::db::NEW);
	auto transaction = db->NewTransaction();

	std::string key = std::to_string(seq);
	seq ++;

	caffe2::TensorProtos protos;
	auto dataProto = protos.add_protos();
	auto labelProto = protos.add_protos();

	dataProto->set_data_type(caffe2::TensorProto::BYTE);
	dataProto->add_dims(dataDim1);
	dataProto->add_dims(dataDim2);

	labelProto->set_data_type(caffe2::TensorProto::INT32);
	labelProto->add_dims(labelDim);

//	cout << "Input data size: " << inputData.size() << endl;
//	for (int i = 0; i < inputData.size(); i ++) {
//		std::cout << inputData.data()[i] << ", ";
//		if ((i + 1) % 16 == 0) {
//			cout << endl;
//		}
//	}
//	cout << endl;
	int testData[4];
	for (int i = 0; i < 4; i ++) {
		testData[i] = i;
	}

//	dataProto->set_byte_data((void*)inputData.data(), inputData.size() * sizeof(int32_t));
	dataProto->set_byte_data(testData, 4 * sizeof(int));
	labelProto->add_int32_data(inputLabel.Get(0));

	std::string dbContent;
	protos.SerializeToString(&dbContent);

	transaction->Put(key, dbContent);
	transaction->Commit();

	std::cout << "Data saved " << std::endl;
//	db->Close();
}

void testRead() {
	std::unique_ptr<DB> testDb(caffe2::db::CreateDB(
			dbType, outputDbPath, caffe2::db::READ));
	std::unique_ptr<Cursor> cursor(testDb->NewCursor());

	if (cursor->Valid()) {
		std::cout << "record key: " << cursor->key() << std::endl;

		caffe2::TensorProtos protos;
		protos.ParseFromString(cursor->value());
		cout << "Protos " << protos.protos_size() << endl;

		for (int protoIndex = 0; protoIndex < protos.protos_size(); protoIndex ++) {
			auto proto = protos.protos(protoIndex);
			cout << "Proto dim size: " << proto.dims_size() << endl;
			for (int i = 0; i < proto.dims_size(); i ++) {
				cout << proto.dims(i) << ", ";
			}
			cout << endl;

			auto data = proto.int32_data();
			cout << "Data size " << data.size() << endl;
//			std::vector<int> dataVec(data.data(), data.data() + data.size());
//			for (int i = 0; i < 8; i ++) {
//				cout << data[i] << ", ";
//			}
//			cout << endl;
//			auto tensor = torch::tensor(dataVec);
//			cout << "Tensor dim " << tensor.sizes() << endl;
//			cout << "Tensor cap: " << tensor.numel() << endl;
		}	}
}

void readValue() {
	const std::string dbPath = "/home/zf/workspaces/res/dbs/lmdbcpptest";
	std::unique_ptr<DB> testDb(caffe2::db::CreateDB(
			dbType, dbPath, caffe2::db::READ));
	std::unique_ptr<Cursor> cursor(testDb->NewCursor());
	int64_t num = 0;
	std::vector<int32_t> labelNums(42, 0);

	while (cursor->Valid()) {
//		cursor->Next();
//		std::cout << "Cursor Key " << cursor->key() << std::endl;

		caffe2::TensorProtos protos;
		protos.ParseFromString(cursor->value());
//		cout << "Protos " << protos.protos_size() << endl;

		for (int protoIndex = 0; protoIndex < protos.protos_size(); protoIndex ++) {
			if (protoIndex % 2 == 0) {
				auto dataProto = protos.protos(protoIndex);
//				cout << "Data dim " << endl;
//				for (int i = 0; i < dataProto.dims_size(); i ++) {
//					cout << "" << dataProto.dims(i) << ", ";
//				}
//				cout << endl;

//				cout << "Data: " << endl;
				auto data = dataProto.int32_data();
//				cout << "Data size " << data.size() << endl;
//				data.
//				torch::tensor(data.data());
//				cout << endl;
			} else {
//				cout << "Data proto " << dataProto.DebugString() << endl;
				auto labelProto = protos.protos(protoIndex);
				labelNums[labelProto.int32_data(0)] ++;
//				cout << "Label proto " << labelProto.DebugString() << endl;
			}
//		ParseFromString(cursor->value(), &protos);
		}
		cursor->Next();

		num ++;
	}

	std::cout << "Got record " << num << std::endl;
	std::cout << "Label distribution " << std::endl;
	for (int i = 0; i < labelNums.size(); i ++) {
		std::cout << labelNums[i] << ", ";
		if ((i + 1) % 9 == 0) {
			std::cout << std::endl;
		}
	}
	cout << endl;

	std::vector<float> labelDist(42, 0);
	for (int i = 0; i < labelDist.size(); i ++) {
		labelDist[i] = (float)labelNums[i] / num;
	}
	matplotlibcpp::grid(true);
	matplotlibcpp::subplot(1, 2, 1);
	matplotlibcpp::bar(labelNums);
	matplotlibcpp::subplot(1, 2, 2);
	matplotlibcpp::bar(labelDist);
	matplotlibcpp::show();
}

int main() {
//	test();
//	testRead();
//	createDB();
	readValue();

	return 0;
}
