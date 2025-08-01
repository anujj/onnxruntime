
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/run_options.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26409)
#endif

ORT_API_STATUS_IMPL(OrtApis::CreateRunOptions, _Outptr_ OrtRunOptions** out) {
  API_IMPL_BEGIN
  *out = new OrtRunOptions();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsSetRunLogVerbosityLevel, _Inout_ OrtRunOptions* options, int value) {
  options->run_log_verbosity_level = value;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsSetRunLogSeverityLevel, _Inout_ OrtRunOptions* options, int value) {
  options->run_log_severity_level = value;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsSetRunTag, _Inout_ OrtRunOptions* options, _In_ const char* run_tag) {
  if (run_tag)
    options->run_tag = run_tag;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsGetRunLogVerbosityLevel, _In_ const OrtRunOptions* options, _Out_ int* out) {
  *out = options->run_log_verbosity_level;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsGetRunLogSeverityLevel, _In_ const OrtRunOptions* options, _Out_ int* out) {
  *out = options->run_log_severity_level;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsGetRunTag, _In_ const OrtRunOptions* options, _Out_ const char** out) {
  *out = options->run_tag.c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsSetTerminate, _Inout_ OrtRunOptions* options) {
  options->terminate = true;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsUnsetTerminate, _Inout_ OrtRunOptions* options) {
  options->terminate = false;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::AddRunConfigEntry, _Inout_ OrtRunOptions* options,
                    _In_z_ const char* config_key, _In_z_ const char* config_value) {
  return onnxruntime::ToOrtStatus(options->config_options.AddConfigEntry(config_key, config_value));
}

ORT_API(const char*, OrtApis::GetRunConfigEntry, _In_ const OrtRunOptions* options, _In_z_ const char* config_key) {
  const auto& config_options = options->config_options.GetConfigOptionsMap();
  if (auto it = config_options.find(config_key); it != config_options.end()) {
    return it->second.c_str();
  } else {
    return nullptr;
  }
}

ORT_API_STATUS_IMPL(OrtApis::RunOptionsAddActiveLoraAdapter, _Inout_ OrtRunOptions* options,
                    const _In_ OrtLoraAdapter* adapter) {
  API_IMPL_BEGIN
  auto* lora_adapter = reinterpret_cast<const onnxruntime::lora::LoraAdapter*>(adapter);
  options->active_adapters.push_back(lora_adapter);
  return nullptr;
  API_IMPL_END
}
