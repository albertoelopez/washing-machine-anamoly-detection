#ifndef WASHING_MACHINE_TFLITE_HELPER_H
#define WASHING_MACHINE_TFLITE_HELPER_H

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_profiler.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <cstdio>

#include "model.h"  // Auto-generated from convert_to_C_array.py
#include "config.h"

// TensorFlow Lite globals
namespace {
// Error reporter
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Model and interpreter
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// Input/output tensors
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Memory management
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

#if ENABLE_TFLITE_PROFILING
// Profiler for performance monitoring
tflite::MicroProfiler profiler;
#endif
}  // namespace

// ===== Function Declarations =====
bool setupTensorFlow();
bool runInference(const float* input_features, float* output_score);
void print_tensor_arena_usage();
void log_tensor_info(const TfLiteTensor* tensor, const char* tensor_type);
const char* get_tensor_type_name(TfLiteType type);
void print_tensor_quantization(const TfLiteTensor* tensor);

// ===== TensorFlow Lite Helper Functions =====

/**
 * Get string representation of tensor type
 */
const char* get_tensor_type_name(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType: return "kTfLiteNoType";
    case kTfLiteFloat32: return "kTfLiteFloat32";
    case kTfLiteInt32: return "kTfLiteInt32";
    case kTfLiteUInt8: return "kTfLiteUInt8";
    case kTfLiteInt8: return "kTfLiteInt8";
    case kTfLiteInt64: return "kTfLiteInt64";
    case kTfLiteString: return "kTfLiteString";
    case kTfLiteBool: return "kTfLiteBool";
    case kTfLiteInt16: return "kTfLiteInt16";
    case kTfLiteComplex64: return "kTfLiteComplex64";
    case kTfLiteComplex128: return "kTfLiteComplex128";
    case kTfLiteFloat16: return "kTfLiteFloat16";
    case kTfLiteFloat64: return "kTfLiteFloat64";
    default: return "Unknown";
  }
}

/**
 * Print quantization parameters for a tensor
 */
void print_tensor_quantization(const TfLiteTensor* tensor) {
  if (tensor->quantization.type == kTfLiteAffineQuantization) {
    TfLiteAffineQuantization* params = 
        (TfLiteAffineQuantization*)(tensor->quantization.params);
    if (params) {
      TF_LITE_REPORT_ERROR(error_reporter, "    Scale: %f", params->scale->data[0]);
      TF_LITE_REPORT_ERROR(error_reporter, "    Zero point: %d", params->zero_point->data[0]);
    }
  } else {
    TF_LITE_REPORT_ERROR(error_reporter, "    No quantization parameters");
  }
}

bool setupTensorFlow() {
  // Map the model into a usable data structure
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), 
             "Model version %d does not match TFLite schema version %d",
    return false;
  }

  // This pulls in all the operation implementations we need
  static tflite::AllOpsResolver resolver;

  #if ENABLE_TFLITE_PROFILING
  // Build interpreter with profiler
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE, 
      error_reporter, nullptr, &profiler);
  #else
  // Build interpreter without profiler for smaller footprint
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE, 
      error_reporter);
  #endif
  
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    print_tensor_arena_usage();
    return false;
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  if (!input || !output) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to get input or output tensors");
    return false;
  }

  #ifdef ENABLE_SERIAL_DEBUG
  // Print model and tensor information
  TF_LITE_REPORT_ERROR(error_reporter, "\n=== TensorFlow Lite Model Info ===");
  TF_LITE_REPORT_ERROR(error_reporter, "Tensor arena size: %d bytes", TENSOR_ARENA_SIZE);
  
  // Print input tensor details
  log_tensor_info(input, "Input");
  log_tensor_info(output, "Output");
  
  // Print memory usage
  print_tensor_arena_usage();
  TF_LITE_REPORT_ERROR(error_reporter, "=================================\n");
  #endif

  return true;
}

/**
 * Run inference on the provided input features
 * @param input_features Array of input features (must match model input size)
 * @param output_score Pointer to store the output score
 * @return true if inference was successful, false otherwise
 */
bool runInference(const float* input_features, float* output_score) {
  if (!interpreter || !input || !output || !input_features || !output_score) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invalid parameters for inference");
    return false;
  }

  // Validate input size
  size_t input_size = input->bytes / sizeof(float);
  if (input->dims->size != 1 || input->dims->data[0] != NUM_FEATURES) {
    TF_LITE_REPORT_ERROR(error_reporter, 
                        "Input size mismatch: expected %d, got %d", 
                        NUM_FEATURES, input->dims->data[0]);
    return false;
  }

  // Copy input data into the model's input tensor
  // Handle different input tensor types
  switch (input->type) {
    case kTfLiteFloat32:
      for (size_t i = 0; i < input_size; i++) {
        input->data.f[i] = input_features[i];
      }
      break;
    case kTfLiteUInt8:
      // Quantize input from float to uint8
      for (size_t i = 0; i < input_size; i++) {
        float scaled_value = input_features[i] / input->params.scale + input->params.zero_point;
        input->data.uint8[i] = static_cast<uint8_t>(scaled_value);
      }
      break;
    case kTfLiteInt8:
      // Quantize input from float to int8
      for (size_t i = 0; i < input_size; i++) {
        float scaled_value = input_features[i] / input->params.scale + input->params.zero_point;
        input->data.int8[i] = static_cast<int8_t>(scaled_value);
      }
      break;
    default:
      TF_LITE_REPORT_ERROR(error_reporter, "Unsupported input type: %s", 
                          get_tensor_type_name(input->type));
      return false;
  }

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Inference failed with status: %d", invoke_status);
    return false;
  }

  // Read the output (assuming single output with one value)
  // Handle different output tensor types
  switch (output->type) {
    case kTfLiteFloat32:
      *output_score = output->data.f[0];
      break;
    case kTfLiteUInt8:
      // Dequantize output from uint8 to float
      *output_score = (output->data.uint8[0] - output->params.zero_point) * output->params.scale;
      break;
    case kTfLiteInt8:
      // Dequantize output from int8 to float
      *output_score = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
      break;
    default:
      TF_LITE_REPORT_ERROR(error_reporter, "Unsupported output type: %s", 
                          get_tensor_type_name(output->type));
      return false;
  }
  
  #if ENABLE_TFLITE_PROFILING && defined(ENABLE_SERIAL_DEBUG)
  // Print profiling information if enabled
  TF_LITE_REPORT_ERROR(error_reporter, "Inference time: %d us", 
                      profiler.GetTotalTicks() * 1000000 / ticks_per_second());
  profiler.Clear();
  #endif

  return true;
}

/**
 * Log detailed information about a tensor
 * @param tensor The tensor to log information about
 * @param tensor_type A string describing the tensor type (e.g., "Input", "Output")
 */
void log_tensor_info(const TfLiteTensor* tensor, const char* tensor_type) {
  if (!tensor) {
    TF_LITE_REPORT_ERROR(error_reporter, "%s tensor is null", tensor_type);
    return;
  }

  // Print basic tensor info
  TF_LITE_REPORT_ERROR(error_reporter, "%s tensor name: %s", tensor_type, 
                      tensor->name ? tensor->name : "unnamed");
  
  TF_LITE_REPORT_ERROR(error_reporter, "  Type: %s", 
                      get_tensor_type_name(tensor->type));
  
  // Print tensor shape
  TF_LITE_REPORT_ERROR(error_reporter, "  Dimensions: %d", tensor->dims->size);
  TF_LITE_REPORT_ERROR(error_reporter, "  Shape: [");
  for (int i = 0; i < tensor->dims->size; i++) {
    TF_LITE_REPORT_ERROR(error_reporter, "%s%d", (i > 0) ? ", " : "", tensor->dims->data[i]);
  }
  TF_LITE_REPORT_ERROR(error_reporter, "]");
  
  // Print tensor size in bytes
  size_t tensor_bytes = 1;
  for (int i = 0; i < tensor->dims->size; i++) {
    tensor_bytes *= tensor->dims->data[i];
  }
  TF_LITE_REPORT_ERROR(error_reporter, "  Size: %d elements (%d bytes)", 
                      (int)tensor_bytes, (int)(tensor_bytes * tflite::GetTypeSize(tensor->type)));
  
  // Print quantization parameters if quantized
  if (tensor->quantization.type == kTfLiteAffineQuantization) {
    TF_LITE_REPORT_ERROR(error_reporter, "  Quantization parameters:");
    print_tensor_quantization(tensor);
  }
  
  // Print memory address
  TF_LITE_REPORT_ERROR(error_reporter, "  Data address: %p", tensor->data.data);
}

/**
 * Print detailed information about tensor arena usage
 */
void print_tensor_arena_usage() {
  if (!interpreter) {
    TF_LITE_REPORT_ERROR(error_reporter, "Interpreter not initialized");
    return;
  }
  
  // Get memory usage information
  size_t used_bytes = interpreter->arena_used_bytes();
  size_t total_bytes = interpreter->arena_allocated_bytes();
  float usage_percent = (used_bytes * 100.0f) / total_bytes;
  
  // Print memory usage summary
  TF_LITE_REPORT_ERROR(error_reporter, "\n=== Memory Usage ===");
  TF_LITE_REPORT_ERROR(error_reporter, "  Used: %d bytes", (int)used_bytes);
  TF_LITE_REPORT_ERROR(error_reporter, "  Total: %d bytes", (int)total_bytes);
  TF_LITE_REPORT_ERROR(error_reporter, "  Usage: %.1f%%", usage_percent);
  
  // Print detailed allocation information if available
  #ifdef TENSORFLOW_LITE_ENABLE_ALLOCATION_TRACKING
  tflite::RecordingMicroAllocator* allocator = 
      reinterpret_cast<tflite::RecordingMicroAllocator*>(
          interpreter->allocator());
  if (allocator) {
    TF_LITE_REPORT_ERROR(error_reporter, "\nDetailed allocation breakdown:");
    allocator->PrintAllocations();
  }
  #endif
  
  // Print warning if usage is high
  if (usage_percent > 85.0f) {
    TF_LITE_REPORT_ERROR(error_reporter, 
        "WARNING: Tensor arena usage is high (%.1f%%). Consider increasing TENSOR_ARENA_SIZE.",
        usage_percent);
  } else if (usage_percent < 50.0f) {
    TF_LITE_REPORT_ERROR(error_reporter, 
        "NOTE: Tensor arena usage is low (%.1f%%). You may reduce TENSOR_ARENA_SIZE to save memory.",
        usage_percent);
  }
  
  TF_LITE_REPORT_ERROR(error_reporter, "===================\n");
}

#endif // WASHING_MACHINE_TFLITE_HELPER_H
