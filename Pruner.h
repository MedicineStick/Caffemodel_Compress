#pragma once
#include<string>
#include<map>
#include<vector>
#include<iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "caffe/proto/caffe.pb.h"

typedef google::protobuf::int64 int_64;
typedef std::pair<double, int_64> param;
typedef std::pair<std::string, param> convParam;
typedef std::vector<convParam> convParams;
typedef std::pair<convParam, convParams> record;
typedef record* precord;
typedef std::pair<int, double> atom;
typedef atom* Patom;


#define _RATE_ "rate"
#define _SCALE_ "scale"
#define less(A,B)(*A < *B)
#define min(A,B)(less(A,B))?*A:*B
#define eq(A,B)(!less(A,B)&&!less(B,A))
#define exch(A,B){Patom t = A; A = B; B = t;}
#define compexch(A,B)if(less(B,A)) exch(A,B);

class Pruner
{
public:
	Pruner();
	Pruner(std::string xml_path);
	void start();
	void read_XML(std::string xml_path);
	void import();
	inline void pruning(){
		switch (pruningMode){
		case rate:
			pruningByRate();
		case size:
			pruningBySize();
		
		}
	};
	inline std::string doubleToString(double num)
	{
		char str[256];
		sprintf(str, "%lf", num);
		std::string result = str;
		return result;
	};
	inline std::string intToString(int_64 i){
		std::stringstream stream;
		stream << i;
		return stream.str();
	};

	bool eltwiseCheck(std::string name);
	bool checkIsConv(std::string name);
	void hS(std::vector<atom>* a, int l, int r);
	void fixUp(std::vector<atom>* a, int k);
	void fixDown(std::vector<atom>* a, int k, int N);
	void pruningByRate();
	void pruningConvByRate(const precord r, std::vector<int>* channelNeedPrune);
	void pruningBottomByRate(const precord r, std::vector<int>* channelNeedPrune);
	int writePrototxt(std::string prototxt1, std::string prototxt2);

	void batchNormPruning(::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator iter_, std::vector<int>* channelNeedPrune, int num_);
	void filterPruning(::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator iter_, std::vector<int>* channelNeedPrune, int num_);
	void channelPruning(::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator iter_, std::vector<int>* channelNeedPrune, int num_);

	void pruningBySize();
	inline bool findInt(std::vector<int>::iterator beg, std::vector<int>::iterator end, int ival){
		while (beg != end){
			if (*beg == ival){
				break;
			}
			else{
				++beg;
			}
		}
		if (beg != end){
			return true;
		}
		else
		{
			return false;
		}
	};
	void writeModel();
	~Pruner();

	std::string xml_Path;
	std::string pruning_caffemodel_path;
	std::string pruning_proto_path;
	std::string pruned_caffemodel_path;
	std::string pruned_proto_path;
	std::string txt_proto_path;
	
	enum ConvCalculateMode
	{
		Norm = 8, L1 = 11, L2 = 12
	};
	enum PruningMode
	{
		rate = 0, size = 1
	};
	int convCalculateMode;
	int pruningMode;

private:
	std::vector<convParam> pruning_rate;
	std::vector<convParam> pruning_rate_eltwise;
	boost::property_tree::ptree configure;
	caffe::NetParameter proto;
	std::vector<record> conv;
	std::vector<record> eltwiseConv;
	convParams convNeedRewriteOnPrototxt;
	std::vector<record>::iterator incur;
	::google::protobuf::RepeatedPtrField< caffe::LayerParameter >* layer;
	::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator it;
	std::map<std::string, std::vector<int>> kernel_pruning;
	std::map < std::string, std::vector<int>> channel_pruning;

};
