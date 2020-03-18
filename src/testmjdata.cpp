#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/proto/caffe2_pb.h>
#include <caffe2/proto/caffe2_legacy.pb.h>
#include <caffe2/core/logging.h>
#include <caffe2/core/blob_serialization.h>
#include <torch/torch.h>

#include <c10/util/Registry.h>

#include <iostream>
#include <string>
#include <math.h>
#include <vector>

#include "caffe2/utils/proto_utils.h"

using caffe2::db::Cursor;
using caffe2::db::DB;
using caffe2::db::Transaction;
using caffe2::CaffeDatum;
using caffe2::TensorProto;
using caffe2::TensorProtos;

using std::string;
using std::cout;
using std::endl;

using ::google::protobuf::Message;

const std::string dbType("lmdb");
//const std::string cppDbPath("/home/zf/workspaces/workspace_java/mjpratice_git/Dp4jPractice/lmdbscenetest/");
const string cppDbPath("/home/zf/workspaces/res/dbs/lmdbtest");

void readSceneDb() {
//	const std::string sceneDbPath = "/home/zf/workspaces/res/dbs/lmdbscenetest";
	const std::string sceneDbPath = "/home/zf/workspaces/res/dbs/lmdb5rowscenetest";
	std::unique_ptr<DB> db(caffe2::db::CreateDB(dbType, sceneDbPath, caffe2::db::READ));
	std::unique_ptr<Cursor> cursor(db->NewCursor());

	while (cursor->Valid()) {
		cout << "Cursor key: " << cursor->key() << endl;

		caffe2::TensorProtos protos;
		protos.ParseFromString(cursor->value());
		cout << "Protos " << protos.protos_size() << endl;

		for (int protoIndex = 0; protoIndex < protos.protos_size(); protoIndex ++) {
			auto proto = protos.protos(protoIndex);
			cout << proto.dims_size() << endl;
			for (int i = 0; i < proto.dims_size(); i ++) {
				cout << proto.dims(i) << ", ";
			}
			cout << endl;

			auto data = proto.int32_data();
			cout << "Data size " << data.size() << endl;
			std::vector<int> dataVec(data.data(), data.data() + data.size());
//			for (int i = 0; i < dataVec.size(); i ++) {
//				cout << dataVec[i] << ", ";
//			}
//			cout << endl;
			auto tensor = torch::tensor(dataVec);
			cout << "Tensor dim " << tensor.sizes() << endl;
			cout << "Tensor cap: " << tensor.numel() << endl;
		}

		cursor->Next();
	}

	db->Close();
}

void readValue(const string dbPath) {
	std::unique_ptr<DB> testDb(caffe2::db::CreateDB(
			dbType, dbPath, caffe2::db::READ));
	std::unique_ptr<Cursor> cursor(testDb->NewCursor());

	while (cursor->Valid()) {
//		cursor->Next();
		std::cout << "Cursor Key " << cursor->key() << std::endl;

		caffe2::TensorProtos protos;
		protos.ParseFromString(cursor->value());
		cout << "Protos " << protos.protos_size() << endl;

		for (int protoIndex = 0; protoIndex < protos.protos_size(); protoIndex ++) {
			if (protoIndex % 2 == 0) {
				auto dataProto = protos.protos(protoIndex);
				cout << "Data dim " << endl;
				for (int i = 0; i < dataProto.dims_size(); i ++) {
					cout << "" << dataProto.dims(i) << ", ";
				}
				cout << endl;

//				cout << "Data: " << endl;
				auto data = dataProto.int32_data();
				cout << "Data size " << data.size() << endl;
//				data.
//				torch::tensor(data.data());
//				cout << endl;
			} else {
//				cout << "Data proto " << dataProto.DebugString() << endl;
				auto labelProto = protos.protos(protoIndex);
				cout << "Label proto " << labelProto.DebugString() << endl;
			}
//		ParseFromString(cursor->value(), &protos);
		}
		cursor->Next();
	}

}

void checkDB(const string dbPath) {
	caffe2::db::DBReader reader("lmdb", dbPath);
	std::cout << "DBReader created " << std::endl;

	std::string key;
	std::string value;

	reader.Read(&key, &value);
	std::cout << key << std::endl;
}

const std::string cppCreateDb = "/home/zf/workspaces/res/dbs/lmdbcpptest";

void createDB(const string dbPath) {
	const std::string dbType = "lmdb";
//	const std::string dbName = "CppTensorDb";

	auto db = caffe2::db::CreateDB(dbType, dbPath, caffe2::db::NEW);
	auto transaction = db->NewTransaction();

	const std::string key = "Testtesttesttest";

	caffe2::TensorProtos protos;
	auto dataProto = protos.add_protos();
	auto labelProto = protos.add_protos();

	dataProto->set_data_type(caffe2::TensorProto::BYTE);
	dataProto->add_dims(1);
	dataProto->add_dims(16);

	labelProto->set_data_type(caffe2::TensorProto::INT32);
	labelProto->add_dims(1);
	labelProto->add_int32_data(3);

	cout << "All protos created " << endl;

	int data[4];
	for (int i = 0; i < 4; i ++) {
		data[i] = i;
	}
	cout << "Data prepared " << endl;

	dataProto->set_byte_data(data, 4 * sizeof(int));
	cout << "data injected " << endl;

	string value;
	protos.SerializeToString(&value);
	cout << "Bytes created" << endl;

	transaction->Put(key, value);
	transaction->Commit();
	cout << "trans committed" << endl;

//	db->Close();
}

int main(int argc, char** argv) {
//	readValue(cppDbPath);
	readValue(cppCreateDb);
//	checkDB(cppDbPath);
//	createDB(cppCreateDb);

//	checkDB(cppCreateDb);
//	readSceneDb();
}
