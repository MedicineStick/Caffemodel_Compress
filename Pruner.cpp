#include "Pruner.h"
#include "caffe/util/io.hpp"
#include <cmath>
#include <fstream>

using namespace caffe;
using namespace std;

Pruner::Pruner(){
}

Pruner::Pruner(string xml_path){
	xml_Path = xml_path;

}

Pruner::~Pruner()
{
}

void Pruner::start(){
	read_XML(xml_Path);
	import();
	pruning();
	writePrototxt(pruning_proto_path, pruned_proto_path);
	writeModel();

}

void Pruner::read_XML(string xml_path){
	read_xml(xml_path, configure);
	boost::property_tree::ptree layers = configure.get_child("filterpruning");
	pruning_caffemodel_path = configure.get<string>("caffemodelpath");
	pruning_proto_path = configure.get<string>("protopath");
	pruned_caffemodel_path = configure.get<string>("prunedcaffemodelpath");
	pruned_proto_path = configure.get<string>("prunedprotopath");
	txt_proto_path = configure.get<string>("txtprotopath");
	pruningMode = atoi(configure.get<string>("PruningMode.mode").c_str());
	convCalculateMode = atoi(configure.get<string>("ConvCalculateMode.mode").c_str());
	
	ReadProtoFromBinaryFile(pruning_caffemodel_path, &proto);
	layer = proto.mutable_layer();
	for (boost::property_tree::ptree::iterator it1 = layers.begin(); it1 != layers.end(); it1++){
		boost::property_tree::ptree clayers = it1->second;
		string name = clayers.get<string>("<xmlattr>.name");
		if (!checkIsConv(name)){
			break;
		}
		string cut = clayers.get<string>("<xmlattr>.cut");
		double rate = atof(cut.c_str());
		for (it = layer->begin(); it != layer->end(); it++){
			if (name == it->name()){
				pruning_rate.push_back(convParam(name, param(rate, it->blobs(0).shape().dim(0))));
				convNeedRewriteOnPrototxt.push_back(convParam(it->name(), param(rate, it->blobs(0).shape().dim(0))));
				break;
			}
		}
	}
}

void Pruner::import(){
	it = layer->begin();
	vector<convParam>::iterator iter1;
	iter1 = pruning_rate.begin();
	while (iter1 != pruning_rate.end()){
		it = layer->begin();
		double rate = iter1->second.first;
		convParams b1;
		string prunedConvName = iter1->first;
		string poolName = "konglusen";
		for (; it != layer->end(); it++){
			string n = it->name();
			if (it->bottom_size() != 0){
				for (int i = 0; i < it->bottom_size(); i++){
					if (prunedConvName == it->bottom(i)){
						if (it->type() == "Convolution"){
							if (prunedConvName == it->bottom(0)){
								b1.push_back(convParam(it->name(), param(rate, it->blobs(0).shape().dim(0))));
								break;
							}
						}
						else if (it->type() == "ConvolutionDepthwise"){

							if (prunedConvName == it->bottom(0)){
								b1.push_back(convParam(it->name(), param(rate, it->blobs(0).shape().dim(0))));
								convNeedRewriteOnPrototxt.push_back(convParam(it->name(), param(rate, it->blobs(0).shape().dim(0))));
								break;
							}
						}

						else if (it->type() == "Pooling"){
							poolName = it->name();
							for (::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator it1 = it; it1 != layer->end(); it1++){
								if (it1->type() == "Convolution" || it1->type() == "ConvolutionDepthwise"){
									if (it1->bottom(0).find(poolName) != string::npos){
										b1.push_back(convParam(it1->name(), param(rate, it1->blobs(0).shape().dim(0))));
									}
								}
							}
						}
					}
				}
			}
			else{
				continue;
			}

		}
		conv.push_back(record(*iter1, b1));
		iter1++;
	}
}

void Pruner::pruningByRate(){

	incur = conv.begin();
	
	for (int i = 0; i < conv.size(); i++){
		vector<int> channelNeedPrune;
		pruningConvByRate(&conv.at(i), &channelNeedPrune);
		pruningBottomByRate(&conv.at(i), &channelNeedPrune);
	}
}

void Pruner::pruningBySize(){

}

void Pruner::hS(vector<atom>* a, int l, int r){
	int k;
	int N = r - l + 1;
	for (k = N / 2; k >= 1; k--){
		fixDown(a, k, N);
	}
	while (N > 1){
		swap(a->at(1), a->at(N));
		fixDown(a, 1, --N);
	}
}

void Pruner::fixUp(vector<atom>* a, int k){
	while (k > 1 && less(&a->at(k / 2).second, &a->at(k).second)){
		swap(a->at(k), a->at(k / 2));
		k = k / 2;
	}
}

void Pruner::fixDown(vector<atom>* a, int k, int N){
	int j;
	while (2 * k <= N){
		j = 2 * k;
		if (j < N && less(&a->at(j).second, &a->at(j + 1).second)){
			j++;
		}
		if (!less(&a->at(k).second, &a->at(j).second)){
			break;
		}
		swap(a->at(k), a->at(j));
		k = j;
	}
}

void Pruner::writeModel(){
	WriteProtoToTextFile(proto, txt_proto_path);
	WriteProtoToBinaryFile(proto, pruned_caffemodel_path);
}

//bool Pruner::eltwiseCheck(string name){
//	for (::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator it1 = layer->begin(); it1 != layer->end(); it1++){
//		if (it1->bottom_size() != 0){
//			for (int i = 0; i < it1->bottom_size(); i++){
//				if ("Eltwise" == it1->type()){
//					if (it1->bottom(i).find("split") != string::npos){
//						if (it1->bottom(i).find(name) != string::npos){
//							return false;
//						}
//					}
//				}
//			}
//		}
//	}
//	return true;
//}

bool Pruner::checkIsConv(string name){
	int count = 0;
	for (::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator it1 = layer->begin(); it1 != layer->end(); it1++){
		if (it1->name() == name)
			if (it1->type() == "Convolution")
				count++;
	}
	return (count == 1) ? true : false;
}

void Pruner::pruningConvByRate(const precord r, vector<int>* pchannelNeedPrune){

	for (it = layer->begin(); it != layer->end(); it++){
		if (r->first.first == it->name()){
			std::vector<atom> convlayervalue;
			convlayervalue.push_back(make_pair(-1, 1));
			int num = it->blobs(0).shape().dim(0);
			int channels = it->blobs(0).shape().dim(1);
			int height = it->blobs(0).shape().dim(2);
			int width = it->blobs(0).shape().dim(3);
			int count = channels * width * height;
			int cutNum = (r->first.second.second)*(r->first.second.first);

			//Modifying the kernel heap by traveling through the computed-average-kernel's -size then sort 
			BlobProto blobData = it->blobs(0);
			double maxData = 0.0;
			int k = blobData.data_size();
			
			switch (convCalculateMode)
			{
			case Pruner::Norm:
				for (int i = 0; i < blobData.data_size(); i++){
					if (maxData < abs(blobData.data(i))){
						maxData = abs(blobData.data(i));
					}
				}
				for (int i = 0; i < num; i++){
					double value = 0.0;
					for (int j = 0; j < count; j++){
						value += abs(blobData.data(i*count + j)) / maxData;
					}
					atom a = make_pair(i, value / count);
					convlayervalue.push_back(a);
				}
				hS(&convlayervalue, 1, num);
				for (int i = 0; i < cutNum; i++){
					pchannelNeedPrune->push_back(convlayervalue.at(i + 1).first);
				}
				break;
			case Pruner::L1:
				for (int i = 0; i < num; i++){
					double value = 0.0;
					for (int j = 0; j < count; j++){
						value += abs(blobData.data(i*count + j));
					}
					atom a = make_pair(i, value);
					convlayervalue.push_back(a);
				}
				hS(&convlayervalue, 1, num);
				for (int i = 0; i < cutNum; i++){
					pchannelNeedPrune->push_back(convlayervalue.at(i + 1).first);
				}
				break;
			case Pruner::L2:
				for (int i = 0; i < num; i++){
					double value = 0.0;
					for (int j = 0; j < count; j++){
						value += (blobData.data(i*count + j))*(blobData.data(i*count + j));
					}
					atom a = make_pair(i, value);
					convlayervalue.push_back(a);
				}
				hS(&convlayervalue, 1, num);
				for (int i = 0; i < cutNum; i++){
					pchannelNeedPrune->push_back(convlayervalue.at(i + 1).first);
				}
				break;
			default:
				break;
			}			

			//start prune
			filterPruning(it, pchannelNeedPrune, num);

			//BatchNorm pruning and Scale Pruning
			it++;
			if (it->type() == "BatchNorm"){
				batchNormPruning(it, pchannelNeedPrune, num);
			}
			break;
			//end
		}
	}
}

void Pruner::pruningBottomByRate(const precord r, vector<int>* pchannelNeedPrune){
	//preform pruning on next layer 
	int num = r->first.second.second;
	int cutNum = (r->first.second.second)*(r->first.second.first);
	int i1 = r->second.size();
	for (int k = 0; k < r->second.size(); k++){
		convParam conv1 = r->second[k];
		string n = conv1.first;
		for (it = layer->begin(); it != layer->end(); it++){
			if (it->name() == conv1.first){
				if (it->type() == "Convolution"){
					channelPruning(it, pchannelNeedPrune, num);
					break;
				}
				else if (it->type() == "ConvolutionDepthwise"){
					filterPruning(it, pchannelNeedPrune, num);
					it++;

					//start prune batchNorm and Scale layer
					if (it->type() == "BatchNorm"){
						batchNormPruning(it, pchannelNeedPrune, num);

					}
					while (it->type() != "Convolution"){
						it++;
					}

					//start prune pointwise conv layer next by depthwiseConv
					string name1 = it->name();
					if (it->type() == "Convolution"){
						channelPruning(it, pchannelNeedPrune, num);
					}

					break;
				}
			}

		}

	}
}

void Pruner::batchNormPruning(::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator iter_, std::vector<int>* pchannelNeedPrune, int num_){
	vector<int>::iterator beg = pchannelNeedPrune->begin();
	vector<int>::iterator end = pchannelNeedPrune->end();
	int cutNum = pchannelNeedPrune->size();
	BlobProto *bnBlob0_ = it->mutable_blobs(0);
	BlobProto bnBlob0 = it->blobs(0);
	bnBlob0_->clear_data();
	for (int j = 0; j < num_; j++){
		if (!findInt(beg, end, j)){
			bnBlob0_->add_data(bnBlob0.data(j));
		}
	}
	BlobShape shape0;
	shape0.add_dim(num_ - cutNum);
	bnBlob0_->mutable_shape()->CopyFrom(shape0);

	BlobProto *bnBlob1_ = it->mutable_blobs(1);
	BlobProto bnBlob1 = it->blobs(1);
	bnBlob1_->clear_data();
	for (int j = 0; j < num_; j++){
		if (!findInt(beg, end, j)){
			bnBlob1_->add_data(bnBlob1.data(j));
		}
	}
	BlobShape shape1;
	shape1.add_dim(num_ - cutNum);
	bnBlob1_->mutable_shape()->CopyFrom(shape1);
	it++;

	BlobProto *sBlob0_ = it->mutable_blobs(0);
	BlobProto sBlob0 = it->blobs(0);
	sBlob0_->clear_data();
	for (int j = 0; j < num_; j++){
		if (!findInt(beg, end, j)){
			sBlob0_->add_data(sBlob0.data(j));
		}
	}
	BlobShape shape2;
	shape2.add_dim(num_ - cutNum);
	sBlob0_->mutable_shape()->CopyFrom(shape2);

	BlobProto *sBlob1_ = it->mutable_blobs(1);
	BlobProto sBlob1 = it->blobs(1);
	sBlob1_->clear_data();
	for (int j = 0; j < num_; j++){
		if (!findInt(beg, end, j)){
			sBlob1_->add_data(sBlob1.data(j));
		}
	}
	BlobShape shape3;
	shape3.add_dim(num_ - cutNum);
	sBlob1_->mutable_shape()->CopyFrom(shape3);

}

void Pruner::filterPruning(::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator iter_, std::vector<int>* pchannelNeedPrune, int num_){
	int channels = it->blobs(0).shape().dim(1);
	int height = it->blobs(0).shape().dim(2);
	int width = it->blobs(0).shape().dim(3);
	int count = channels * width * height;
	int cutNum = pchannelNeedPrune->size();
	BlobProto *blob_ = it->mutable_blobs(0);
	BlobProto blob = it->blobs(0);
	blob_->clear_data();
	vector<int>::iterator beg = pchannelNeedPrune->begin();
	vector<int>::iterator end = pchannelNeedPrune->end();
	for (int j = 0; j < num_; j++){
		if (!findInt(beg, end, j)){
			for (int g = 0; g < count; g++){
				blob_->add_data(blob.data(j*count + g));
			}
		}
	}
	BlobShape shape;
	shape.add_dim(num_ - cutNum);
	shape.add_dim(channels);
	shape.add_dim(height);
	shape.add_dim(width);
	blob_->mutable_shape()->CopyFrom(shape);

	// We will perform bias update based if the bias of conv existed.
	if (it->blobs_size() > 1){
		BlobProto *blob_ = it->mutable_blobs(1);
		BlobProto blob = it->blobs(1);
		blob_->clear_data();
		for (int j = 0; j < num_; j++){
			if (!findInt(beg, end, j)){
				blob_->add_data(blob.data(j));
			}
		}
		BlobShape shape;
		shape.add_dim(num_ - cutNum);
		blob_->mutable_shape()->CopyFrom(shape);
	}

	it->mutable_convolution_param()->set_num_output(num_ - cutNum);
}

void Pruner::channelPruning(::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator iter_, std::vector<int>* pchannelNeedPrune, int num_){
	int nextLayKerNum = iter_->blobs(0).shape().dim(0);
	int nextLayChannel = iter_->blobs(0).shape().dim(1);
	int nextLayKerH = iter_->blobs(0).shape().dim(2);
	int nextLayKerW = iter_->blobs(0).shape().dim(3);
	int cutNum = pchannelNeedPrune->size();
	int nextLayKerCount = (num_ - cutNum) * nextLayKerH * nextLayKerW;
	int counts = nextLayChannel * nextLayKerH * nextLayKerW;
	int dimSize = nextLayKerH * nextLayKerW;
	BlobProto *blob1_ = iter_->mutable_blobs(0);
	BlobProto blob1 = iter_->blobs(0);
	blob1_->clear_data();
	vector<int>::iterator beg = pchannelNeedPrune->begin();
	vector<int>::iterator end = pchannelNeedPrune->end();
	for (int j = 0; j < nextLayKerNum; j++){
		for (int g = 0; g < nextLayChannel; g++){
			if (!findInt(beg, end, g)){
				//blob1_->add_data(j * nextLayKerCount + g);
				for (int m = 0; m < dimSize; m++){
					blob1_->add_data(blob1.data(j * counts + g * dimSize + m));
				}
			}
		}
	}
	BlobShape shape1;
	shape1.add_dim(nextLayKerNum);
	shape1.add_dim(num_ - cutNum);
	shape1.add_dim(nextLayKerH);
	shape1.add_dim(nextLayKerW);
	blob1_->mutable_shape()->CopyFrom(shape1);

}

int  Pruner::writePrototxt(std::string prototxt1, std::string prototxt2){
	std::fstream fin_in(pruning_proto_path, std::ios::in | std::ios::binary);
	std::fstream fin_out(pruned_proto_path, std::ios::out | std::ios::binary);
	if (!fin_in || !fin_out)
		return 0;
	string str1 = "name";
	string str2 = "num_output";
	string str3 = "type";
	string str;
	string nametemp;
	bool final_flag = false;
	bool nor_flag = false;
	int prunedNum;
	while (getline(fin_in, str)){
		if (str.find("prob") != -1){
			final_flag = true;
		}
		if (final_flag == true){
			fin_out << str << '\n';
			continue;
		}
		int index = -1;
		if (str.find(str1) != -1){
			for (auto& r : convNeedRewriteOnPrototxt){
				string s = '"' +  r.first + '"';
				index = str.find(s);
				if (index != -1){
					int num = r.second.second;
					int cut = r.second.first*r.second.second;
					prunedNum = num - cut;
					nor_flag = true;
					break;
				}
			}
		}
		if (str.find(str2) != -1){
			if (!nor_flag){
				fin_out << str << '\n';
			}
			else{
				fin_out << "    num_output: " + std::to_string(prunedNum) << '\n';
				nor_flag = false;

			}
		}
		else{
			fin_out << str << '\n';

		}
	}
	return 1;
}
