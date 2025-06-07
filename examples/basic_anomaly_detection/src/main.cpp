#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <driver/rtc_io.h>
#include <esp_sleep.h>
#include <esp_task_wdt.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Include your model header (generated from .tflite)
#include "model.h"

// Configuration - Update these based on your hardware and requirements
namespace Config {
    // Hardware pins
    constexpr int kLedPin = LED_BUILTIN;     // Built-in LED for status
    constexpr int kAlertPin = 4;             // Alert indicator LED
    constexpr int kI2CSdaPin = 21;           // I2C SDA pin
    constexpr int kI2CSclPin = 22;           // I2C SCL pin
    
    // Model and inference settings
    constexpr int kTensorArenaSize = 16 * 1024;  // 16KB tensor arena (adjust based on model size)
    constexpr int kNumSamples = 128;            // Number of samples per inference
    constexpr int kNumFeatures = 6;              // Number of features per sample (3 accel + 3 gyro)
    constexpr float kAnomalyThreshold = 0.7f;   // Threshold for anomaly detection (0.0 to 1.0)
    
    // Timing and power management
    constexpr uint64_t kDeepSleepUs = 10 * 1000 * 1000;  // 10 seconds deep sleep between inferences
    constexpr uint32_t kSampleIntervalMs = 20;          // 20ms between samples (~50Hz)
    constexpr uint32_t kWdtTimeout = 30;                // Watchdog timeout in seconds
    
    // Debug settings
    constexpr bool kEnableSerialDebug = true;
    constexpr uint32_t kSerialBaudRate = 115200;
    constexpr bool kEnableModelDetails = true;
}

// Global variables
namespace {
    // TensorFlow Lite globals
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    
    // Memory for tensors and intermediate buffers
    alignas(16) static uint8_t tensor_arena[Config::kTensorArenaSize];
    
    // Sensor data buffer [kNumSamples][kNumFeatures]
    static float input_buffer[Config::kNumSamples * Config::kNumFeatures];
    
    // System state
    bool is_initialized = false;
    uint32_t inference_count = 0;
    uint32_t anomaly_count = 0;
    
    // Timing
    uint32_t last_sample_time = 0;
    uint32_t last_inference_time = 0;
    uint32_t last_serial_output = 0;
}

// Forward declarations
void setup();
void loop();
void setupPins();
void setupI2C();
void setupSensors();
bool setupTensorFlow();
void readSensors();
bool detectAnomaly();
void handleAnomaly(bool is_anomaly);
void printDebugInfo();
void enterDeepSleep();
void printTensorDetails(const TfLiteTensor* tensor, const char* name);
void printMemoryStats();

// Error handling
#define CHECK_TFLITE_ERROR(x) \
    do { \
        TfLiteStatus status = (x); \
        if (status != kTfLiteOk) { \
            TF_LITE_REPORT_ERROR(error_reporter, \
                               "Error at %s:%d: %d", \
                               __FILE__, __LINE__, status); \
            return false; \
        } \
    } while (0)

void setup() {
    // Initialize serial communication
    if (Config::kEnableSerialDebug) {
        Serial.begin(Config::kSerialBaudRate);
        while (!Serial && millis() < 2000) {
            // Wait for serial port to connect or timeout after 2 seconds
        }
    }
    
    // Configure watchdog timer
    if (esp_task_wdt_init(Config::kWdtTimeout, true) != ESP_OK) {
        if (Config::kEnableSerialDebug) {
            Serial.println("Failed to initialize watchdog timer");
        }
    }
    esp_task_wdt_add(NULL); // Add current thread to WDT watch
    
    // Setup hardware
    setupPins();
    setupI2C();
    setupSensors();
    
    // Initialize TensorFlow Lite
    if (!setupTensorFlow()) {
        if (Config::kEnableSerialDebug) {
            Serial.println("Failed to initialize TensorFlow Lite");
        }
        // Blink error pattern
        while (true) {
            digitalWrite(Config::kLedPin, HIGH);
            delay(100);
            digitalWrite(Config::kLedPin, LOW);
            delay(100);
        }
    }
    
    // Initialization complete
    is_initialized = true;
    if (Config::kEnableSerialDebug) {
        Serial.println("\n=== Setup Complete ===");
        Serial.println("System initialized and ready for anomaly detection");
        Serial.printf("Tensor arena size: %d bytes\n", Config::kTensorArenaSize);
        Serial.printf("Sample rate: %d Hz\n", 1000 / Config::kSampleIntervalMs);
        Serial.println("=====================\n");
    }
    
    // Indicate ready state
    for (int i = 0; i < 3; i++) {
        digitalWrite(Config::kLedPin, HIGH);
        delay(100);
        digitalWrite(Config::kLedPin, LOW);
        delay(100);
    }
}

void loop() {
    // Feed the watchdog
    esp_task_wdt_reset();
    
    // Check if system is initialized
    if (!is_initialized) {
        if (Config::kEnableSerialDebug) {
            Serial.println("System not initialized, waiting...");
        }
        delay(1000);
        return;
    }
    
    uint32_t current_time = millis();
    
    // Collect sensor data at regular intervals
    if (current_time - last_sample_time >= Config::kSampleIntervalMs) {
        last_sample_time = current_time;
        readSensors();
        
        // Check if we have enough samples for inference
        static size_t sample_count = 0;
        sample_count++;
        
        if (sample_count >= Config::kNumSamples) {
            sample_count = 0;
            
            // Perform inference
            bool is_anomaly = detectAnomaly();
            handleAnomaly(is_anomaly);
            
            // Update counters and timers
            inference_count++;
            last_inference_time = current_time;
            
            // Print debug info periodically
            if (Config::kEnableSerialDebug && 
                (current_time - last_serial_output >= 5000 || last_serial_output == 0)) {
                printDebugInfo();
                last_serial_output = current_time;
            }
            
            // Enter deep sleep between inference cycles
            if (Config::kDeepSleepUs > 0) {
                if (Config::kEnableSerialDebug) {
                    Serial.println("Entering deep sleep...");
                    Serial.flush();
                }
                enterDeepSleep();
            }
        }
    }
    
    // Blink status LED to show activity
    static uint32_t last_led_toggle = 0;
    if (current_time - last_led_toggle >= 500) {  // Blink every 500ms
        digitalWrite(Config::kLedPin, !digitalRead(Config::kLedPin));
        last_led_toggle = current_time;
    }
}

void setupPins() {
    // Configure GPIO pins
    pinMode(Config::kLedPin, OUTPUT);
    pinMode(Config::kAlertPin, OUTPUT);
    digitalWrite(Config::kLedPin, LOW);
    digitalWrite(Config::kAlertPin, LOW);
    
    // Configure any other required pins here
    // For example, if using SPI or other GPIOs
}

void setupI2C() {
    // Initialize I2C communication
    Wire.begin(Config::kI2CSdaPin, Config::kI2CSclPin);
    Wire.setClock(400000);  // 400kHz I2C clock
    
    if (Config::kEnableSerialDebug) {
        Serial.println("I2C initialized");
    }
}

void setupSensors() {
    // Initialize sensors here
    // Example for MPU6050 (uncomment and modify as needed):
    /*
    if (!mpu.begin()) {
        if (Config::kEnableSerialDebug) {
            Serial.println("Failed to initialize MPU6050");
        }
        while (1) {
            // Halt on sensor initialization failure
            delay(1000);
        }
    }
    
    // Configure sensor ranges and rates
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    */
    
    if (Config::kEnableSerialDebug) {
        Serial.println("Sensors initialized");
    }
}

bool setupTensorFlow() {
    // Set up error reporting
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    
    // Map the model into a usable data structure
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
                          "Model schema version %d not equal "
                          "to supported version %d.",
                          model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    
    // This pulls in all the operation implementations we need
    static tflite::AllOpsResolver resolver;
    
    // Build an interpreter to run the model
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, Config::kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate memory from the tensor_arena for the model's tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return false;
    }
    
    // Obtain pointers to the model's input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Print model and tensor information if debug is enabled
    if (Config::kEnableModelDetails && Config::kEnableSerialDebug) {
        Serial.println("\n=== Model Information ===");
        Serial.printf("Input tensor size: %d\n", input->bytes);
        Serial.printf("Input dimensions: %dD [", input->dims->size);
        for (int i = 0; i < input->dims->size; i++) {
            Serial.printf("%d", input->dims->data[i]);
            if (i < input->dims->size - 1) Serial.print(" x ");
        }
        Serial.println("]");
        
        Serial.printf("Output tensor size: %d\n", output->bytes);
        Serial.printf("Output dimensions: %dD [", output->dims->size);
        for (int i = 0; i < output->dims->size; i++) {
            Serial.printf("%d", output->dims->data[i]);
            if (i < output->dims->size - 1) Serial.print(" x ");
        }
        Serial.println("]");
        
        // Print memory usage
        printMemoryStats();
        Serial.println("=========================\n");
    }
    
    if (Config::kEnableSerialDebug) {
        Serial.println("TensorFlow Lite initialized successfully");
    }
    
    return true;
}

void readSensors() {
    static size_t sample_index = 0;
    
    // Read from sensors and store in the circular buffer
    // Example for MPU6050 (uncomment and modify as needed):
    /*
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    
    // Store in circular buffer
    size_t idx = (sample_index % Config::kNumSamples) * Config::kNumFeatures;
    
    // Accelerometer data (m/s²)
    input_buffer[idx + 0] = a.acceleration.x;
    input_buffer[idx + 1] = a.acceleration.y;
    input_buffer[idx + 2] = a.acceleration.z;
    
    // Gyroscope data (rad/s)
    input_buffer[idx + 3] = g.gyro.x;
    input_buffer[idx + 4] = g.gyro.y;
    input_buffer[idx + 5] = g.gyro.z;
    
    // Temperature (if needed)
    // float temperature = temp.temperature;
    */
    
    // For now, simulate sensor data
    for (int i = 0; i < Config::kNumFeatures; i++) {
        size_t idx = (sample_index % Config::kNumSamples) * Config::kNumFeatures + i;
        input_buffer[idx] = random(100) / 100.0f;  // Random values between 0 and 1
    }
    
    sample_index++;
    
    // Print debug info for the first few samples
    if (Config::kEnableSerialDebug && sample_index <= Config::kNumSamples) {
        Serial.printf("Sample %d: ", sample_index);
        for (int i = 0; i < Config::kNumFeatures; i++) {
            Serial.printf("%.2f ", input_buffer[(sample_index-1) * Config::kNumFeatures + i]);
        }
        Serial.println();
    }
}

bool detectAnomaly() {
    // Check if model is loaded
    if (!interpreter || !input || !output) {
        if (Config::kEnableSerialDebug) {
            Serial.println("Model not properly initialized");
        }
        return false;
    }
    
    // Check input tensor dimensions
    if (input->type != kTfLiteFloat32) {
        if (Config::kEnableSerialDebug) {
            Serial.println("Input tensor type not supported");
        }
        return false;
    }
    
    // Copy input data to the model's input tensor
    // The model expects input in the shape [1, kNumSamples, kNumFeatures]
    float* input_data = input->data.f;
    memcpy(input_data, input_buffer, Config::kNumSamples * Config::kNumFeatures * sizeof(float));
    
    // Run inference
    uint32_t start_time = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    uint32_t inference_time = micros() - start_time;
    
    if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Inference failed");
        return false;
    }
    
    // Get the output value (assuming single output with anomaly score)
    float anomaly_score = output->data.f[0];
    
    // Log the inference results
    if (Config::kEnableSerialDebug) {
        Serial.printf("\n=== Inference Results ===\n");
        Serial.printf("Inference time: %u us\n", inference_time);
        Serial.printf("Anomaly score: %.6f\n", anomaly_score);
        Serial.printf("Threshold: %.6f\n", Config::kAnomalyThreshold);
        Serial.printf("Anomaly detected: %s\n", 
                     (anomaly_score > Config::kAnomalyThreshold) ? "YES" : "NO");
        Serial.println("========================\n");
    }
    
    // Return true if anomaly is detected
    return anomaly_score > Config::kAnomalyThreshold;
}

void handleAnomaly(bool is_anomaly) {
    // Update anomaly counter if anomaly is detected
    if (is_anomaly) {
        anomaly_count++;
        
        // Visual feedback
        digitalWrite(Config::kAlertPin, HIGH);
        
        if (Config::kEnableSerialDebug) {
            Serial.println("ALERT: Anomaly detected!");
        }
        
        // Additional actions on anomaly detection
        // e.g., log to SD card, send notification, etc.
    } else {
        // Normal operation
        digitalWrite(Config::kAlertPin, LOW);
    }
}

void printDebugInfo() {
    if (!Config::kEnableSerialDebug) return;
    
    uint32_t current_time = millis();
    uint32_t uptime_sec = current_time / 1000;
    
    Serial.println("\n=== System Status ===");
    Serial.printf("Uptime: %02d:%02d:%02d\n", 
                 (uptime_sec / 3600) % 24,  // Hours
                 (uptime_sec / 60) % 60,    // Minutes
                 uptime_sec % 60);          // Seconds
    
    Serial.printf("Inference count: %u\n", inference_count);
    Serial.printf("Anomaly count: %u (%.1f%%)\n", 
                 anomaly_count, 
                 (inference_count > 0) ? (100.0f * anomaly_count / inference_count) : 0.0f);
    
    // Print memory statistics
    printMemoryStats();
    
    Serial.println("====================\n");
}

void enterDeepSleep() {
    // Turn off peripherals to save power
    digitalWrite(Config::kLedPin, LOW);
    digitalWrite(Config::kAlertPin, LOW);
    
    // Configure wake-up sources
    esp_sleep_enable_timer_wakeup(Config::kDeepSleepUs);
    
    // Enter deep sleep
    esp_deep_sleep_start();
    // Code will restart from setup() after wake-up
}

void printTensorDetails(const TfLiteTensor* tensor, const char* name) {
    if (!tensor || !name) return;
    
    Serial.printf("Tensor '%s' type: %d, bytes: %d, dims: %d [", 
                 name, tensor->type, tensor->bytes, tensor->dims->size);
    
    for (int i = 0; i < tensor->dims->size; i++) {
        Serial.printf("%d", tensor->dims->data[i]);
        if (i < tensor->dims->size - 1) Serial.print(" x ");
    }
    Serial.println("]");
}

void printMemoryStats() {
    // Print memory usage statistics
    extern int __heap_start, *__brkval;
    int v;
    int free_memory = (int) &v - (__brkval == 0 ? (int) &__heap_start : (int) __brkval);
    
    Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
    Serial.printf("Min free heap: %d bytes\n", ESP.getMinFreeHeap());
    
    // Tensor arena usage
    size_t used_arena = interpreter->arena_used_bytes();
    size_t total_arena = Config::kTensorArenaSize;
    float arena_usage_pct = 100.0f * used_arena / total_arena;
    
    Serial.printf("Tensor arena: %u/%u bytes (%.1f%%)\n", 
                 used_arena, total_arena, arena_usage_pct);
    
    if (arena_usage_pct > 90.0f) {
        Serial.println("WARNING: Tensor arena usage is high!");
    }
}
