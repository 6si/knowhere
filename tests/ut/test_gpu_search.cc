// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <atomic>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

#ifdef KNOWHERE_WITH_CUVS

template <typename T>
void
check_search(const int64_t nb, const int64_t nq, const int64_t dim, const int64_t seed, std::string name, int version,
             const knowhere::Json& conf, float min_recall = 0.9f) {
    auto train_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(nb, dim, seed));
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(nq, dim, seed + 2));

    if (std::is_same_v<T, knowhere::fp16> &&
        (name == knowhere::IndexEnum::INDEX_GPU_IVFFLAT || name == knowhere::IndexEnum::INDEX_CUVS_IVFFLAT)) {
        // IVF-FLAT FP16 distances become too large, so we normalize the dataset
        // https://github.com/rapidsai/cuvs/issues/914
        knowhere::NormalizeDataset<T>(train_ds);
        knowhere::NormalizeDataset<T>(query_ds);
    }

    auto idx = knowhere::IndexFactory::Instance().Create<T>(name, version).value();

    // 1. Self-search
    auto res = idx.Build(train_ds, conf);
    REQUIRE(res == knowhere::Status::success);
    auto results = idx.Search(train_ds, conf, nullptr);
    REQUIRE(results.has_value());

    auto ids = results.value()->GetIds();
    for (int i = 1; i < nq; ++i) {
        CHECK(ids[i] == i);
    }

    // 2. Search a query dataset

    results = idx.Search(query_ds, conf, nullptr);
    REQUIRE(results.has_value());

    auto gt = knowhere::BruteForce::Search<T>(train_ds, query_ds, conf, nullptr);
    REQUIRE(gt.has_value());

    float recall = GetKNNRecall(*gt.value(), *results.value());
    REQUIRE(recall >= min_recall);
}

TEST_CASE("Test All GPU Index", "[search]") {
    using Catch::Approx;

    int64_t nb = 10000, nq = 1000;
    int64_t dim = 128;
    int64_t seed = 42;

    auto version = GenTestVersionList();

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        json[knowhere::meta::TOPK] = 1;
        json[knowhere::meta::RADIUS] = 10.0;
        json[knowhere::meta::RANGE_FILTER] = 0.0;
        return json;
    };

    auto bruteforce_gen = base_gen;

    auto ivfflat_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 20;
        json[knowhere::indexparam::NPROBE] = 18;
        return json;
    };

    auto ivfpq_gen = [ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::M] = 0;
        json[knowhere::indexparam::NBITS] = 8;
        return json;
    };

    auto cagra_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::INTERMEDIATE_GRAPH_DEGREE] = 64;
        json[knowhere::indexparam::GRAPH_DEGREE] = 32;
        json[knowhere::indexparam::ITOPK_SIZE] = 128;
        return json;
    };

    auto cagra_hnsw_gen = [](auto&& upstream_gen) {
        return [upstream_gen]() {
            knowhere::Json json = upstream_gen();
            json[knowhere::indexparam::ADAPT_FOR_CPU] = false;
            json[knowhere::indexparam::EF] = 128;
            return json;
        };
    };

    auto refined_gen = [](auto&& upstream_gen) {
        return [upstream_gen]() {
            knowhere::Json json = upstream_gen();
            json[knowhere::indexparam::REFINE_RATIO] = 1.5;
            json[knowhere::indexparam::CACHE_DATASET_ON_DEVICE] = true;
            return json;
        };
    };

    auto cosine_gen = [](auto&& upstream_gen) {
        return [upstream_gen]() {
            knowhere::Json json = upstream_gen();
            json[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
            return json;
        };
    };
    auto hamming_gen = [](auto&& upstream_gen) {
        return [upstream_gen]() {
            knowhere::Json json = upstream_gen();
            json[knowhere::meta::METRIC_TYPE] = knowhere::metric::HAMMING;
            json[knowhere::indexparam::BUILD_ALGO] = "ITERATIVE";
            return json;
        };
    };

    SECTION("Test Gpu Index Search") {
        using std::make_tuple;
        auto [name, gen, min_recall] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>, float>({
            make_tuple(knowhere::IndexEnum::INDEX_GPU_BRUTEFORCE, bruteforce_gen, 0.999f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_IVFFLAT, ivfflat_gen, 0.95f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_IVFFLAT, refined_gen(ivfflat_gen), 0.95f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_IVFPQ, ivfpq_gen, 0.75f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_IVFPQ, refined_gen(ivfpq_gen), 0.75f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_CAGRA, cagra_gen, 0.9f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_CAGRA, cagra_hnsw_gen(cagra_gen), 0.9f),
        }));

        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json conf = knowhere::Json::parse(cfg_json);
        conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        if (knowhere::IndexFactory::Instance().FeatureCheck(name, knowhere::feature::FLOAT32)) {
            check_search<knowhere::fp32>(nb, nq, dim, seed, name, version, conf, min_recall);
        }
        if (knowhere::IndexFactory::Instance().FeatureCheck(name, knowhere::feature::FP16)) {
            check_search<knowhere::fp16>(nb, nq, dim, seed, name, version, conf, min_recall);
        }
        if (knowhere::IndexFactory::Instance().FeatureCheck(name, knowhere::feature::INT8)) {
            check_search<knowhere::int8>(nb, nq, dim, seed, name, version, conf, min_recall);
        }
    }

    SECTION("Test Gpu Index Search With Bitset") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, bruteforce_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, refined_gen(ivfflat_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, refined_gen(ivfpq_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
            GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
        const auto bitset_percentages = {0.4f, 0.98f};
        for (const float percentage : bitset_percentages) {
            for (const auto& gen_func : gen_bitset_funcs) {
                auto bitset_data = gen_func(nb, percentage * nb);
                knowhere::BitsetView bitset(bitset_data.data(), nb);
                auto results = idx.Search(query_ds, json, bitset);
                REQUIRE(results.has_value());
                auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, bitset);
                float recall = GetKNNRecall(*gt.value(), *results.value());
                if (percentage == 0.98f) {
                    REQUIRE(recall > 0.4f);
                } else {
                    REQUIRE(recall > 0.7f);
                }
            }
        }
    }

    SECTION("Test Gpu Index Search TopK") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, bruteforce_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, refined_gen(ivfflat_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, refined_gen(ivfpq_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_hnsw_gen(cagra_gen)),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        const auto topk_values = {// Tuple with [TopKValue, Threshold]
                                  make_tuple(5, 0.85f), make_tuple(25, 0.85f), make_tuple(100, 0.85f)};

        for (const auto& topKTuple : topk_values) {
            json[knowhere::meta::TOPK] = std::get<0>(topKTuple);
            auto results = idx.Search(query_ds, json, nullptr);
            REQUIRE(results.has_value());
            auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, nullptr);
            float recall = GetKNNRecall(*gt.value(), *results.value());
            REQUIRE(recall >= std::get<1>(topKTuple));
        }
    }

    SECTION("Test Gpu Index Serialize/Deserialize") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, bruteforce_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, refined_gen(ivfflat_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, refined_gen(ivfpq_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_hnsw_gen(cagra_gen)),
        }));

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);
        auto idx_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        idx_.Deserialize(bs);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        auto results = idx_.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        // Due to issues with the filtering of invalid values, index 0 is temporarily not being checked.
        for (int i = 1; i < nq; ++i) {
            CHECK(ids[i] == i);
        }
    }

    SECTION("Test Gpu Index Cagra Adapt For Cpu") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);
        auto idx_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        json[knowhere::indexparam::ADAPT_FOR_CPU] = true;
        json[knowhere::indexparam::EF] = 128;
        idx_.Deserialize(bs, json);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        auto results = idx_.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        // Due to issues with the filtering of invalid values, index 0 is temporarily not being checked.
        for (int i = 1; i < nq; ++i) {
            CHECK(ids[i] == i);
        }
    }

    SECTION("Test Gpu Index Cagra Adapt For Cpu Without Ef") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);
        auto idx_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        knowhere::Json deser_json = json;
        deser_json[knowhere::indexparam::ADAPT_FOR_CPU] = true;
        // Intentionally NOT setting ef — this is the bug scenario
        idx_.Deserialize(bs, deser_json);
        // Search without ef in params — should not crash
        auto results = idx_.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        for (int i = 1; i < nq; ++i) {
            CHECK(ids[i] == i);
        }
    }

    SECTION("Test Gpu Index Search Simple Bitset") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>(
            {make_tuple(knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, bruteforce_gen),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, refined_gen(ivfflat_gen)),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, ivfpq_gen),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, refined_gen(ivfpq_gen)),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen)}));
        auto rows = 64;
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        if (name == knowhere::IndexEnum::INDEX_CUVS_CAGRA) {
            json[knowhere::indexparam::INTERMEDIATE_GRAPH_DEGREE] = 32;
            json[knowhere::indexparam::GRAPH_DEGREE] = 32;
            json[knowhere::indexparam::ITOPK_SIZE] = 32;
        }
        auto train_ds = GenDataSet(rows, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        std::vector<uint8_t> bitset_data(8);
        bitset_data[0] = 0b10100010;
        bitset_data[1] = 0b00100011;
        bitset_data[2] = 0b10100010;
        bitset_data[3] = 0b00100111;
        bitset_data[4] = 0b10100000;
        bitset_data[5] = 0b00000000;
        bitset_data[6] = 0b00000010;
        bitset_data[7] = 0b11100011;
        knowhere::BitsetView bitset(bitset_data.data(), rows);
        auto results = idx.Search(train_ds, json, bitset);
        REQUIRE(results.has_value());
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, train_ds, json, bitset);
        // Go through the results and check if the id is in the bitset
        for (int i = 0; i < rows; ++i) {
            auto id = results.value()->GetIds()[i];
            if (id == -1) {
                continue;
            }
            REQUIRE(!(bitset_data[id / 8] & (1 << (id % 8))));
        }
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.8f);
    }

    SECTION("Test Gpu Index Search Cosine Metric") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>(
            {make_tuple(knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, cosine_gen(bruteforce_gen)),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, cosine_gen(ivfflat_gen)),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, cosine_gen(ivfpq_gen)),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, cosine_gen(refined_gen(ivfpq_gen))),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cosine_gen(cagra_gen))}));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 1);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, nullptr);
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall > 0.65f);
    }

    SECTION("Test Gpu Index Search Hamming Metric") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>(
            {make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, hamming_gen(cagra_gen))}));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::bin1>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        nb = 1500;  // Reduce dataset size to have less distance = 0 when testing query distance
        auto train_ds = GenBinDataSet(nb, dim, seed);
        auto query_ds = GenBinDataSet(nq, dim, seed + 1);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.Count() == nb);
        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto gt = knowhere::BruteForce::Search<knowhere::bin1>(train_ds, query_ds, json, nullptr);
        auto dist = results.value()->GetDistance();
        auto gt_dist = gt.value()->GetDistance();
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall > 0.8f);
        recall = GetKNNRelativeRecall(*gt.value(), *results.value(), true);
        REQUIRE(recall > 0.95f);
        for (int i = 1; i < nq; ++i) {
            // Check query distance
            CHECK(GetRelativeLoss(gt_dist[i], dist[i]) < 0.1f);
        }
    }

    // GPU_HNSW tests: build on CPU as HNSW, serialize, deserialize as GPU_HNSW, search on GPU.
    SECTION("Test GPU HNSW Search (CPU build -> GPU search)") {
        // Build a CPU HNSW index
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        // Serialize the CPU index
        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        // Deserialize as GPU_HNSW
        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        // Count()/Dim() must survive the CPU-copy release after GPU upload;
        // regression guard for returning -1 (segment treated as empty).
        REQUIRE(gpu_idx.Count() == nb);
        REQUIRE(gpu_idx.Dim() == dim);

        // Search on GPU
        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        // Compare against brute force
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.9f);
    }

    SECTION("Test GPU HNSW Search Cosine Metric") {
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 1);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // GPU HNSW cosine recall tracks L2 (measured ~0.98 on L40S); keep the
        // floor in line with the L2/IP sections so real regressions are caught.
        REQUIRE(recall >= 0.80f);
    }

    SECTION("Test GPU HNSW Search TopK") {
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        const auto topk_values = {
            std::make_tuple(5, 0.85f),
            std::make_tuple(25, 0.85f),
            std::make_tuple(100, 0.85f),
        };

        for (const auto& [topk, threshold] : topk_values) {
            hnsw_json[knowhere::meta::TOPK] = topk;
            auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
            REQUIRE(results.has_value());
            auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
            float recall = GetKNNRecall(*gt.value(), *results.value());
            REQUIRE(recall >= threshold);
        }
    }

    SECTION("Test GPU HNSW Serialize/Deserialize Round Trip") {
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        // Self-search: each vector should find itself as nearest neighbor.
        // The query set is train_ds (nb rows), so check all nb results — using
        // nq only exercised the first nq of nb vectors.
        auto results = gpu_idx.Search(train_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        int correct = 0;
        for (int i = 0; i < nb; ++i) {
            if (ids[i] == i)
                correct++;
        }
        float self_recall = static_cast<float>(correct) / nb;
        REQUIRE(self_recall >= 0.95f);
    }

    SECTION("Test GPU HNSW SQ8 Deserialization") {
        // Build a CPU HNSW_SQ index, then load as GPU_HNSW
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version)
                           .value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        // GPU_HNSW should accept HNSW_SQ binaries
        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // SQ8 has lower recall than flat due to quantization
        REQUIRE(recall >= 0.7f);
    }

    SECTION("Test GPU HNSW INT8 Deserialization") {
        // Build a CPU int8 HNSW (HNSW_SQ QT_8bit_direct_signed storage), then
        // load and search on GPU. int8 vectors stay in their native 1-byte
        // layout on the device (upload_int8_dataset) and are up-converted to
        // fp32 per element in the search kernel. This exercises the native
        // signed-int8 device path, distinct from the unsigned SQ8 test above
        // which routes through the decode-to-fp32 upload branch.
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(GenDataSet(nb, dim, seed));
        auto query_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(GenDataSet(nq, dim, seed + 2));

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::int8>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::int8>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        auto gt = knowhere::BruteForce::Search<knowhere::int8>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // int8 native path should match the CPU int8 HNSW recall closely.
        REQUIRE(recall >= 0.9f);
    }

    SECTION("Test GPU HNSW FP16 Deserialization") {
        // Build a CPU HNSW (fp16 -> HNSW_SQ QT_fp16 storage), then load and
        // search on GPU. fp16 vectors stay in their native 2-byte layout on the
        // device and are up-converted to fp32 per element in the search kernel.
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(GenDataSet(nb, dim, seed));
        auto query_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(GenDataSet(nq, dim, seed + 2));

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp16>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp16>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        auto gt = knowhere::BruteForce::Search<knowhere::fp16>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // fp16 is near-lossless (10-bit mantissa); recall should stay close to
        // the fp32 flat path.
        REQUIRE(recall >= 0.9f);
    }

    SECTION("Test GPU HNSW BF16 Deserialization") {
        // Build a CPU HNSW (bf16 -> HNSW_SQ QT_bf16 storage), then load and
        // search on GPU. bf16 vectors stay in their native 2-byte layout on the
        // device and are up-converted to fp32 per element in the search kernel.
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(GenDataSet(nb, dim, seed));
        auto query_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(GenDataSet(nq, dim, seed + 2));

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::bf16>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::bf16>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        auto gt = knowhere::BruteForce::Search<knowhere::bf16>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // bf16 has a 7-bit mantissa (coarser than fp16) but a wide exponent;
        // recall stays high but with a slightly looser floor.
        REQUIRE(recall >= 0.85f);
    }
}

TEST_CASE("Test CPU vs GPU HNSW Comparison", "[gpu_hnsw_compare]") {
    using Catch::Approx;

    int64_t nb = 10000, nq = 100;
    int64_t dim = 128;
    int64_t seed = 42;

    auto version = GenTestVersionList();

    auto run_comparison = [&](const std::string& metric, float min_recall, float max_dist_drift) {
        CAPTURE(metric, nb, nq, dim);

        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = metric;
        hnsw_json[knowhere::meta::TOPK] = 10;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        // --- Build CPU HNSW ---
        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        // Serialize for GPU deserialization
        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        // --- CPU Search with timing ---
        auto cpu_start = std::chrono::high_resolution_clock::now();
        auto cpu_results = cpu_idx.Search(query_ds, hnsw_json, nullptr);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        REQUIRE(cpu_results.has_value());
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // --- Build GPU HNSW from serialized CPU index ---
        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        // Warm-up search (first GPU call has kernel launch overhead)
        auto warmup = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(warmup.has_value());

        // --- GPU Search with timing ---
        auto gpu_start = std::chrono::high_resolution_clock::now();
        auto gpu_results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        REQUIRE(gpu_results.has_value());
        double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        // --- Brute force ground truth ---
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());

        // --- Recall comparison ---
        float cpu_recall = GetKNNRecall(*gt.value(), *cpu_results.value());
        float gpu_recall = GetKNNRecall(*gt.value(), *gpu_results.value());

        // --- Distance accuracy: compare GPU distances against CPU distances ---
        auto cpu_dist = cpu_results.value()->GetDistance();
        auto gpu_dist = gpu_results.value()->GetDistance();
        auto k = hnsw_json[knowhere::meta::TOPK].get<int64_t>();

        double total_rel_error = 0.0;
        int valid_pairs = 0;
        for (int64_t i = 0; i < nq * k; ++i) {
            if (std::abs(cpu_dist[i]) > 1e-6f) {
                total_rel_error += std::abs((gpu_dist[i] - cpu_dist[i]) / cpu_dist[i]);
                valid_pairs++;
            }
        }
        double avg_rel_dist_error = (valid_pairs > 0) ? (total_rel_error / valid_pairs) : 0.0;

        // --- ID overlap: how many of the same results are returned ---
        auto cpu_ids = cpu_results.value()->GetIds();
        auto gpu_ids = gpu_results.value()->GetIds();
        int id_overlap = 0;
        for (int64_t q = 0; q < nq; ++q) {
            std::set<int64_t> cpu_set(cpu_ids + q * k, cpu_ids + (q + 1) * k);
            for (int64_t j = 0; j < k; ++j) {
                if (cpu_set.count(gpu_ids[q * k + j])) {
                    id_overlap++;
                }
            }
        }
        float id_overlap_ratio = static_cast<float>(id_overlap) / (nq * k);

        // --- Print comparison report ---
        fprintf(stderr, "\n=== CPU vs GPU HNSW Comparison (%s, nb=%ld, nq=%ld, dim=%ld, k=%ld) ===\n", metric.c_str(),
                (long)nb, (long)nq, (long)dim, (long)k);
        fprintf(stderr, "  CPU recall@%ld: %.4f\n", (long)k, cpu_recall);
        fprintf(stderr, "  GPU recall@%ld: %.4f\n", (long)k, gpu_recall);
        fprintf(stderr, "  Recall delta (GPU - CPU): %+.4f\n", gpu_recall - cpu_recall);
        fprintf(stderr, "  CPU search time: %.2f ms\n", cpu_ms);
        fprintf(stderr, "  GPU search time: %.2f ms (includes H2D/D2H transfers)\n", gpu_ms);
        fprintf(stderr, "  Speedup: %.2fx\n", cpu_ms / gpu_ms);
        fprintf(stderr, "  Avg relative distance error: %.6f\n", avg_rel_dist_error);
        fprintf(stderr, "  ID overlap (CPU vs GPU): %.4f (%d/%ld)\n", id_overlap_ratio, id_overlap, (long)(nq * k));
        fprintf(stderr, "===\n\n");

        // --- Assertions ---
        // GPU recall should be close to CPU recall (within a small tolerance)
        REQUIRE(gpu_recall >= min_recall);
        // GPU should return at least 80% of the same IDs as CPU
        REQUIRE(id_overlap_ratio >= 0.8f);
        // Average relative distance error should be small
        REQUIRE(avg_rel_dist_error <= max_dist_drift);
    };

    SECTION("L2 metric comparison") {
        run_comparison(knowhere::metric::L2, 0.85f, 0.05);
    }

    SECTION("IP metric comparison") {
        run_comparison(knowhere::metric::IP, 0.85f, 0.05);
    }

    SECTION("COSINE metric comparison") {
        run_comparison(knowhere::metric::COSINE, 0.60f, 0.10);
    }

    SECTION("Varying ef comparison") {
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 10;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());

        fprintf(stderr, "\n=== ef sweep (L2, nb=%ld, nq=%ld, k=10) ===\n", (long)nb, (long)nq);
        fprintf(stderr, "  %6s  %10s  %10s  %10s  %10s\n", "ef", "cpu_recall", "gpu_recall", "cpu_ms", "gpu_ms");

        for (int ef : {16, 32, 64, 128, 256, 512}) {
            hnsw_json[knowhere::indexparam::EF] = ef;

            auto t0 = std::chrono::high_resolution_clock::now();
            auto cpu_res = cpu_idx.Search(query_ds, hnsw_json, nullptr);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto gpu_res = gpu_idx.Search(query_ds, hnsw_json, nullptr);
            auto t2 = std::chrono::high_resolution_clock::now();

            REQUIRE(cpu_res.has_value());
            REQUIRE(gpu_res.has_value());

            float cpu_recall = GetKNNRecall(*gt.value(), *cpu_res.value());
            float gpu_recall = GetKNNRecall(*gt.value(), *gpu_res.value());
            double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double gpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

            fprintf(stderr, "  %6d  %10.4f  %10.4f  %10.2f  %10.2f\n", ef, cpu_recall, gpu_recall, cpu_ms, gpu_ms);

            // GPU recall should be reasonable at all ef values
            REQUIRE(gpu_recall >= 0.5f);
        }
        fprintf(stderr, "===\n\n");
    }
}

// Regression tests for the three Codex P1 findings on the gpu-hnsw-faiss branch:
//   #1 quantized-cosine inverse-norm semantics (native low-precision kernels +
//      SQ8/FP16/BF16 cosine parity),
//   #2 search-vs-reload lifetime safety,
//   #3 phase-separated load-resource accounting.
TEST_CASE("Test GPU HNSW Codex P1 Regressions", "[gpu_hnsw_p1]") {
    using Catch::Approx;

    const int64_t nb = 4000, nq = 200;
    const int64_t dim = 64;
    const int64_t seed = 42;
    const auto version = GenTestVersionList();

    auto sq_json = [&](const std::string& metric, const std::string& sq_type, int64_t topk) {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::HNSW_M] = 16;
        json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        json[knowhere::indexparam::EF] = 200;
        json[knowhere::indexparam::SQ_TYPE] = sq_type;
        return json;
    };

    // Build an HNSW_SQ index on the fp32 (real, non-mock) node so the serialized
    // storage carries the requested SQ qtype. Deserializing that as GPU_HNSW on
    // the fp32 node selects the native device representation from the storage
    // qtype (FP16 -> half kernel, BF16 -> __nv_bfloat16 kernel, SQ8 -> fp32
    // decode). This bypasses the int8/fp16/bf16 IndexNodeDataMockWrapper, which
    // otherwise converts data to fp32 and routes the fp32 kernel.

    // #1a: native low-precision (FP16/BF16) and SQ8 device paths — L2.
    SECTION("Native low-precision device path (L2): FP16 / BF16 / SQ8") {
        const auto sq_type = GENERATE(std::string("fp16"), std::string("bf16"), std::string("sq8"));
        CAPTURE(sq_type);
        auto json = sq_json(knowhere::metric::L2, sq_type, 10);

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version)
                           .value();
        REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // Lossy SQ paths; a conservative floor. fp16/bf16 are near-lossless, sq8
        // coarser.
        REQUIRE(recall >= (sq_type == "sq8" ? 0.65f : 0.85f));
    }

    // #1b: quantized-cosine parity. The discriminating input is vectors that
    // share directions but have widely different magnitudes: cosine must ignore
    // magnitude. The CPU cosine index records inverse norms from the ORIGINAL
    // input vectors; the GPU path must upload those same norms rather than
    // recomputing them from lossily-decoded codes. With the pre-fix behavior
    // (recompute-from-decoded / normalize-decoded) the SQ8 scores and top-1 IDs
    // diverge from the CPU index. Assert per-query top-1 ID and per-result score,
    // not just aggregate recall.
    SECTION("Quantized cosine parity (CPU vs GPU): SQ8 / FP16 / BF16") {
        const auto sq_type = GENERATE(std::string("sq8"), std::string("fp16"), std::string("bf16"));
        CAPTURE(sq_type);
        const int64_t topk = 10;
        auto json = sq_json(knowhere::metric::COSINE, sq_type, topk);

        // Directions from a fixed RNG, then scale each row by a magnitude that
        // spans three orders of magnitude so magnitude cannot be ignored by luck.
        auto make_scaled = [&](int64_t rows, int64_t s) {
            auto ds = GenDataSet(rows, dim, s);
            auto* data = const_cast<float*>(static_cast<const float*>(ds->GetTensor()));
            std::mt19937 rng(static_cast<uint32_t>(s + 7));
            std::uniform_real_distribution<float> mag(0.5f, 500.0f);
            for (int64_t r = 0; r < rows; ++r) {
                float scale = mag(rng);
                for (int64_t d = 0; d < dim; ++d) {
                    data[r * dim + d] *= scale;
                }
            }
            return ds;
        };
        auto train_ds = make_scaled(nb, seed);
        auto query_ds = make_scaled(nq, seed + 2);

        auto cpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version)
                           .value();
        REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
        auto cpu_results = cpu_idx.Search(query_ds, json, nullptr);
        REQUIRE(cpu_results.has_value());

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);
        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);
        auto gpu_results = gpu_idx.Search(query_ds, json, nullptr);
        REQUIRE(gpu_results.has_value());

        const auto* cpu_ids = cpu_results.value()->GetIds();
        const auto* gpu_ids = gpu_results.value()->GetIds();
        const auto* cpu_dist = cpu_results.value()->GetDistance();
        const auto* gpu_dist = gpu_results.value()->GetDistance();

        // Top-1 must match the CPU index for the overwhelming majority of
        // queries: identical cosine semantics (same norms) means identical
        // nearest neighbor modulo rare quantization ties.
        int top1_match = 0;
        for (int64_t q = 0; q < nq; ++q) {
            if (cpu_ids[q * topk] == gpu_ids[q * topk]) {
                top1_match++;
            }
        }
        float top1_ratio = static_cast<float>(top1_match) / nq;
        CAPTURE(top1_ratio);
        REQUIRE(top1_ratio >= 0.95f);

        // Per-result cosine scores must agree closely (both decode identical SQ
        // codes and apply the SAME original inverse norms). A loose floor would
        // hide the pre-fix norm mismatch, so keep the tolerance tight.
        int compared = 0, close = 0;
        for (int64_t q = 0; q < nq; ++q) {
            // Only compare positions where CPU and GPU agree on the id, so score
            // comparison is like-for-like.
            for (int64_t j = 0; j < topk; ++j) {
                if (cpu_ids[q * topk + j] == gpu_ids[q * topk + j]) {
                    compared++;
                    if (GetRelativeLoss(cpu_dist[q * topk + j], gpu_dist[q * topk + j]) < 0.02f) {
                        close++;
                    }
                }
            }
        }
        REQUIRE(compared > 0);
        float close_ratio = static_cast<float>(close) / compared;
        CAPTURE(close_ratio);
        REQUIRE(close_ratio >= 0.98f);
    }

    // #2: search must not use-after-free a GPU index that a concurrent reload
    // resets/rebuilds. Search continuously from several threads while another
    // thread repeatedly Deserialize()s (reloads) the same index. Run under
    // TSAN/ASAN to catch the lifetime race the atomic<shared_ptr> snapshot fixes.
    SECTION("Search vs concurrent reload (lifetime safety)") {
        auto json = sq_json(knowhere::metric::L2, "fp16", 10);
        json.erase(knowhere::indexparam::SQ_TYPE);  // plain fp32 HNSW for the reload driver
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

        std::atomic<bool> stop{false};
        std::atomic<bool> failed{false};
        std::vector<std::thread> searchers;
        for (int t = 0; t < 6; ++t) {
            searchers.emplace_back([&]() {
                while (!stop.load(std::memory_order_relaxed)) {
                    auto r = gpu_idx.Search(query_ds, json, nullptr);
                    // A search may race a reload window; success or a benign
                    // "not ready" error are both acceptable, a crash is not.
                    if (!r.has_value() && r.error() != knowhere::Status::empty_index &&
                        r.error() != knowhere::Status::cuda_runtime_error) {
                        failed.store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
        std::thread reloader([&]() {
            for (int i = 0; i < 30 && !stop.load(std::memory_order_relaxed); ++i) {
                if (gpu_idx.Deserialize(bs) != knowhere::Status::success) {
                    failed.store(true, std::memory_order_relaxed);
                }
            }
            stop.store(true, std::memory_order_relaxed);
        });
        reloader.join();
        stop.store(true, std::memory_order_relaxed);
        for (auto& th : searchers) {
            th.join();
        }
        REQUIRE(!failed.load());
        // Index is still usable after the reload storm.
        auto after = gpu_idx.Search(query_ds, json, nullptr);
        REQUIRE(after.has_value());
    }

    // Codex coverage #3: more than four simultaneous searches (the device scratch
    // pool has four slots), with varying nq/k/ef, each validated against its own
    // brute-force ground truth. Exercises scratch-pool contention/serialization.
    SECTION("More than four concurrent searches with varying nq/k/ef") {
        auto build_json = sq_json(knowhere::metric::L2, "fp16", 10);
        build_json.erase(knowhere::indexparam::SQ_TYPE);
        auto train_ds = GenDataSet(nb, dim, seed);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(cpu_idx.Build(train_ds, build_json) == knowhere::Status::success);
        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);
        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

        struct Case {
            int64_t nq;
            int64_t k;
            int ef;
        };
        const std::vector<Case> cases = {{50, 5, 32},   {120, 10, 64}, {200, 20, 128}, {80, 50, 256},
                                         {30, 10, 512}, {150, 1, 48},  {64, 100, 300}, {90, 25, 96}};
        std::atomic<bool> ok{true};
        std::vector<std::thread> threads;
        for (const auto& c : cases) {
            threads.emplace_back([&, c]() {
                knowhere::Json j;
                j[knowhere::meta::DIM] = dim;
                j[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
                j[knowhere::meta::TOPK] = c.k;
                j[knowhere::indexparam::EF] = c.ef;
                auto q = GenDataSet(c.nq, dim, seed + 100 + c.ef);
                auto r = gpu_idx.Search(q, j, nullptr);
                if (!r.has_value()) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, q, j, nullptr);
                if (!gt.has_value() || GetKNNRecall(*gt.value(), *r.value()) < 0.7f) {
                    ok.store(false, std::memory_order_relaxed);
                }
            });
        }
        for (auto& th : threads) {
            th.join();
        }
        REQUIRE(ok.load());
    }

    // #3: phase-separated load-resource accounting. GPU_HNSW frees its CPU copy
    // after uploading to VRAM, so retained memoryCost is 0 while the transient
    // peak (maxMemoryCost) must be non-zero and cover the download buffer +
    // deserialized CPU index + decode/graph staging. This estimate is a static
    // computation and needs no live GPU.
    SECTION("Load-resource estimate: memoryCost=0, maxMemoryCost covers peak") {
        knowhere::Json params;
        params[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        params[knowhere::meta::DIM] = dim;

        const uint64_t file_size = static_cast<uint64_t>(nb) * dim * sizeof(float);
        auto gpu_res = knowhere::IndexStaticFaced<knowhere::fp32>::EstimateLoadResource(
            knowhere::IndexEnum::INDEX_GPU_HNSW, version, file_size, nb, dim, params);
        REQUIRE(gpu_res.has_value());
        // Retained host memory is ~0 after the CPU copy is freed.
        REQUIRE(gpu_res.value().memoryCost == 0);
        REQUIRE(gpu_res.value().diskCost == 0);
        // Transient peak is non-zero and at least covers the fp32 decode staging
        // plus the download buffer (2*file_size + rows*dim*4 in the estimator).
        REQUIRE(gpu_res.value().maxMemoryCost > 0);
        const uint64_t decode_staging = static_cast<uint64_t>(nb) * dim * sizeof(float);
        REQUIRE(gpu_res.value().maxMemoryCost >= file_size + decode_staging);

        // A plain CPU HNSW keeps its data resident: memoryCost is non-zero and it
        // does not opt into the peak field (maxMemoryCost stays 0 -> Milvus falls
        // back to its 2*memoryCost heuristic).
        auto cpu_res = knowhere::IndexStaticFaced<knowhere::fp32>::EstimateLoadResource(
            knowhere::IndexEnum::INDEX_HNSW, version, file_size, nb, dim, params);
        REQUIRE(cpu_res.has_value());
        REQUIRE(cpu_res.value().memoryCost > 0);
        REQUIRE(cpu_res.value().maxMemoryCost == 0);
    }
}
#endif
