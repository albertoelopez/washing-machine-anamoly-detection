#ifndef WASHING_MACHINE_TFLITE_HELPER_H
#define WASHING_MACHINE_TFLITE_HELPER_H

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"  // Auto-generated from convert_to_C_array.py
#include "config.h"

namespace {
// TensorFlow Lite globals
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 8 * 1024;  // Adjust based on model size
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setupTensorFlow() {
  // Set up logging
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                        "Model provided is schema version %d not equal "
                        "to supported version %d.",
                        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's input and output tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Check tensor dimensions
  if ((input->dims->size != 2) || (input->dims->data[1] != NUM_FEATURES)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                        "Bad input tensor parameters in model");
    return;
  }

  Serial.println("TensorFlow Lite initialized");
}

float runInference(float* input_features) {
  // Copy input data to the model's input tensor
  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = input_features[i];
  }

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return -1.0f;
  }

  // Get the output value (assuming single output with anomaly score)
  return output->data.f[0];
}

#endif // WASHING_MACHINE_TFLITE_HELPER_H
