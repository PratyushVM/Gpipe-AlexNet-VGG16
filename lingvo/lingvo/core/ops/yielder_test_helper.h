/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef LINGVO_CORE_OPS_YIELDER_TEST_HELPER_H_
#define LINGVO_CORE_OPS_YIELDER_TEST_HELPER_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "lingvo/core/ops/record_yielder.h"

namespace tensorflow {
namespace lingvo {

class MockRecordYielder : public BasicRecordYielder {
 public:
  MockRecordYielder() : BasicRecordYielder() {}
  MOCK_METHOD(Status, Yield, (Record * record), (override));
  MOCK_METHOD(void, Close, (), (override));
  MOCK_METHOD(int64, current_epoch, (), (const, override));
};

// Generates n plain text files with m lines each.
void GeneratePlainTextTestData(const string& prefix, int n, int m);

// Generates checkpoint file and data file for test.
void GenerateCheckpointPlainTextTestData(const string& prefix, int m);
void UpdateCheckpointPlainTextTestData(const string& prefix, int m);

// Computes input source distribution of lines read from plain text data
// generated by GeneratePlainTextTestData  function.
std::unordered_map<std::string, float> ComputeInputSourceDistribution(
    const std::vector<string>& vals);

}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_YIELDER_TEST_HELPER_H_
