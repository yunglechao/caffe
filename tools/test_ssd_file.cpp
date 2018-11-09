#include <cstdio>

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include "caffe/caffe.hpp"
#include "caffe/net.hpp"
#include "caffe/util/bbox_util.hpp"

using namespace std;
using namespace boost;
using namespace caffe;

template<typename Dtype>
int test_ssd_accuracy(int argc, char** argv);

template <class Type>
Type string_to_num(const std::string str);

template <class Type>
Type string_to_num(const std::string str){
  std::istringstream iss(str);
  Type num;
  iss >> num;
  return num;
}

int main(int argc, char** argv) {
  return test_ssd_accuracy<float>(argc, argv);
}

template<typename Dtype>
int test_ssd_accuracy(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  std::map<int, std::map<int, std::vector<std::pair<float, int> > > > all_true_pos;
  std::map<int, std::map<int, std::vector<std::pair<float, int> > > > all_false_pos;
  std::map<int, std::map<int, int> > all_num_pos;

  const int num_required_args = 5;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a SSD newwork with an input data layer and trained weights, and then"
    " compute mAP on specific dataset.\n"
    "Usage: test_ssd_accurary SSD_net_prototxt pretrained_net_param"
    "  ap_version num_mini_batches  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can choose one ap_version from 11point, MaxIntegral, Integral by seting ap_version arg.";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    int device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    caffe::Caffe::SetDevice(device_id);
    caffe::Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    caffe::Caffe::set_mode(Caffe::CPU);
  }

  arg_pos = 0;  // the name of the executable
  std::string ssd_proto(argv[++arg_pos]);
  std::string pretrained_binary_proto(argv[++arg_pos]);
  std::string ap_version(argv[++arg_pos]);
  int num_mini_batches = atoi(argv[++arg_pos]);

  // boost::shared_ptr<Net<Dtype> > test_net(
      // new Net<Dtype>(ssd_proto, caffe::TEST));
  // test_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  Dtype loss = 0;
  int test_iter = num_mini_batches;
  LOG(ERROR) << "Total Iteration: " << test_iter;

  for (int i = 0; i < test_iter; ++i) {
    Dtype iter_loss;
    std::string str;
    std::ostringstream name_string_stream;
    name_string_stream << "/home/zhaoyongle/pkl/output" << i << ".txt";
    const std::string file_name = name_string_stream.str();
    std::ifstream file(file_name.c_str());
    std::vector<float> result_vec;
    while (!file.eof()) {
      std::getline(file, str);
      float num = string_to_num<float>(str);
      result_vec.push_back(num);
    }
    // const std::vector<caffe::Blob<Dtype>*>& result = test_net->Forward(&iter_loss);
    // if (param_.test_compute_loss()) {
      // loss += iter_loss;
    // }
    // LOG(ERROR) << "Iteration: " << i << " Loss: " << iter_loss;
    // LOG(ERROR) << "result size: " << result.size();
    for (int j = 0; j < 1; ++j) {
      int num_det = static_cast<int>(result_vec.size() / 5);
      LOG(ERROR) << "num_det: " << num_det;
      for (int k = 0; k < num_det; ++k) {
        int item_id = static_cast<int>(result_vec[k * 5]);
        int label = static_cast<int>(result_vec[k * 5 + 1]);
        if (item_id == -1) {
          // Special row of storing number of positives for a label.
          if (all_num_pos[j].find(label) == all_num_pos[j].end()) {
            all_num_pos[j][label] = static_cast<int>(result_vec[k * 5 + 2]);
          } else {
            all_num_pos[j][label] += static_cast<int>(result_vec[k * 5 + 2]);
          }
        } else {
          // Normal row storing detection status.
          float score = result_vec[k * 5 + 2];
          int tp = static_cast<int>(result_vec[k * 5 + 3]);
          int fp = static_cast<int>(result_vec[k * 5 + 4]);
          // LOG(ERROR) << "label: " << label << " score: " << score << " tp: " << tp << " fp: " << fp;
          if (tp == 0 && fp == 0) {
            // Ignore such case. It happens when a detection bbox is matched to
            // a difficult gt bbox and we don't evaluate on difficult gt bbox.
            continue;
          }
          all_true_pos[j][label].push_back(std::make_pair(score, tp));
          all_false_pos[j][label].push_back(std::make_pair(score, fp));
        }
      }
    }
  }
  LOG(ERROR) << "Compute mAP";
  for (int i = 0; i < all_true_pos.size(); ++i) {
    if (all_true_pos.find(i) == all_true_pos.end()) {
      LOG(FATAL) << "Missing output_blob true_pos: " << i;
    }
    const std::map<int, std::vector<std::pair<float, int> > >& true_pos =
        all_true_pos.find(i)->second;
    if (all_false_pos.find(i) == all_false_pos.end()) {
      LOG(FATAL) << "Missing output_blob false_pos: " << i;
    }
    const std::map<int, std::vector<std::pair<float, int> > >& false_pos =
        all_false_pos.find(i)->second;
    if (all_num_pos.find(i) == all_num_pos.end()) {
      LOG(FATAL) << "Missing output_blob num_pos: " << i;
    }
    const std::map<int, int>& num_pos = all_num_pos.find(i)->second;
    std::map<int, float> APs;
    float mAP = 0.;
    // Sort true_pos and false_pos with descend scores.
    for (std::map<int, int>::const_iterator it = num_pos.begin();
         it != num_pos.end(); ++it) {
      int label = it->first;
      int label_num_pos = it->second;
      if (true_pos.find(label) == true_pos.end()) {
        LOG(WARNING) << "Missing true_pos for label: " << label;
        continue;
      }
      const std::vector<std::pair<float, int> >& label_true_pos =
          true_pos.find(label)->second;
      if (false_pos.find(label) == false_pos.end()) {
        LOG(WARNING) << "Missing false_pos for label: " << label;
        continue;
      }
      const std::vector<std::pair<float, int> >& label_false_pos =
          false_pos.find(label)->second;
      std::vector<float> prec, rec;
      // LOG(ERROR) << "class" << label << " num_tp: " << label_true_pos.size();
      ComputeAP(label_true_pos, label_num_pos, label_false_pos,
                ap_version, &prec, &rec, &(APs[label]));
      // for (std::vector<float>::const_iterator it = prec.begin();
           // it != prec.end(); ++it) {
        // LOG(ERROR) << "class" << label << " prec: " << *it;
      // }
      // for (std::vector<float>::const_iterator it = rec.begin();
           // it != rec.end(); ++it) {
        // LOG(ERROR) << "class" << label << " rec: " << *it;
      // }
      mAP += APs[label];
      if (true) {//(param_.show_per_class_result()) {
        LOG(ERROR) << "class" << label << ": " << APs[label];
      }
    }
    mAP /= num_pos.size();
    // const int output_blob_index = test_net->output_blob_indices()[i];
    // const std::string& output_name = test_net->blob_names()[output_blob_index];
    LOG(ERROR) << "    Test net output #" << i << ": = "
              << mAP;
  }
  LOG(ERROR)<< "Successfully computed the mAP!";
  return 0;
}
