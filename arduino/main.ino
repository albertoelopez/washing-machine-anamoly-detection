/*
 * Washing Machine Anomaly Detection
 * 
 * This sketch reads sensor data from an ESP32, runs inference using a TensorFlow Lite model,
 * and detects anomalies in washing machine operation.
 * 
 * Features:
 * - Real-time sensor data collection
 * - TensorFlow Lite Micro for on-device ML
 * - Power-efficient operation
 * - Comprehensive error handling
 */

#include <Arduino.h>
#include <Wire.h>
#include <SD.h>
#include <SPI.h>
#include <esp_task_wdt.h>
#include <esp_sleep.h>
#include <driver/adc.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
#include "tflite_helper.h"
#include "sensors.h"
#include "config.h"

// Global variables
RTC_DATA_ATTR static uint32_t boot_count = 0;  // Persists through deep sleep
RTC_DATA_ATTR static uint32_t anomaly_count = 0;
static float features[NUM_FEATURES] = {0};  // Current sensor readings
static float anomaly_scores[ANOMALY_WINDOW] = {0};  // Circular buffer for anomaly scores
static int anomaly_index = 0;  // Current position in the circular buffer
static bool alert_triggered = false;  // Alert state
volatile int anomaly_count = 0;  // Mark as volatile since it's modified in ISR
bool tflite_initialized = false;

// TensorFlow Lite globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create a memory pool for the input, output, and intermediate arrays
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// Timing variables
unsigned long last_sample_time = 0;
unsigned long last_inference_time = 0;
unsigned long last_status_time = 0;
unsigned long last_activity_time = 0;  // Track last user/sensor activity

// Error tracking
int consecutive_errors = 0;
const int MAX_CONSECUTIVE_ERRORS = 5;

void setup() {
  // Increment boot counter
  boot_count++;
  
  // Initialize serial communication
  Serial.begin(115200);
  delay(1000);  // Give serial port time to initialize
  
  // Print boot info
  Serial.println("\n=== Washing Machine Monitor ===");
  Serial.print("Boot count: ");
  Serial.println(boot_count);
  Serial.print("Anomaly count: ");
  Serial.println(anomaly_count);
  
  // Initialize watchdog timer
  esp_task_wdt_init(WDT_TIMEOUT, true);  // Enable panic so ESP32 restarts on WDT
  esp_task_wdt_add(NULL);  // Add current thread to WDT watch
  
  // Initialize hardware
  if (!setupSensors()) {
    Serial.println("Sensor initialization failed!");
    // Don't return, try to continue with limited functionality
  }
  
  // Initialize TensorFlow Lite
  tflite_initialized = setupTensorFlow();
  if (!tflite_initialized) {
    Serial.println("TensorFlow Lite initialization failed!");
    // Don't return, try to continue with limited functionality
  }
  
  // Initialize SD card for data logging
  if (!setupSDCard()) {
    Serial.println("SD card initialization failed. Data logging disabled.");
  }
  
  // Set up built-in LED for alerts
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  
  // Print memory info (ESP32 specific)
  #ifdef ESP32
    Serial.print("Free heap: ");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes");
    Serial.print("Min free heap: ");
    Serial.print(ESP.getMinFreeHeap());
    Serial.println(" bytes");
  #endif
  
  Serial.println("Washing Machine Monitor Ready");
  
  // Reset watchdog timer
  esp_task_wdt_reset();
}

void loop() {
  // Reset watchdog timer
  #ifdef ENABLE_HW_WATCHDOG
  esp_task_wdt_reset();
  #endif
  
  unsigned long current_time = millis();
  bool activity_detected = false;
  
  // 1. Read sensors at fixed interval (100Hz)
  if (current_time - last_sample_time >= SAMPLE_INTERVAL_MS) {
    if (readSensors(features)) {
      // Update last activity time on successful sensor read
      last_activity_time = current_time;
      activity_detected = true;
      consecutive_errors = 0;  // Reset error counter on successful read
    } else {
      consecutive_errors++;
      if (consecutive_errors >= MAX_CONSECUTIVE_ERRORS) {
        enterErrorState("Too many consecutive sensor errors");
      }
    }
    last_sample_time = current_time;
  }
  
  // 2. Run inference at fixed interval (1Hz)
  if (current_time - last_inference_time >= INFERENCE_INTERVAL_MS) {
    if (tflite_initialized) {
      float anomaly_score = 0.0;
      
      if (runInference(features, &anomaly_score)) {
        // Update moving average of anomaly scores
        anomaly_scores[anomaly_index] = anomaly_score;
        anomaly_index = (anomaly_index + 1) % ANOMALY_WINDOW;
        
        // Calculate average anomaly score
        float avg_score = 0.0;
        for (int i = 0; i < ANOMALY_WINDOW; i++) {
          avg_score += anomaly_scores[i];
        }
        avg_score /= ANOMALY_WINDOW;
        
        // Check for anomaly
        if (avg_score > ANOMALY_THRESHOLD) {
          handleAnomaly(avg_score);
          activity_detected = true;  // Count anomaly as activity
        } else {
          alert_triggered = false;
        }
        
        #ifdef ENABLE_SERIAL_DEBUG
        Serial.print("Anomaly score: ");
        Serial.println(avg_score, 4);
        #endif
      }
    } else {
      // Fallback mode: Use simple thresholding on sensor data
      // This is a basic example - adjust thresholds based on your specific sensors
      float accel_mag = sqrt(features[0]*features[0] + features[1]*features[1] + features[2]*features[2]);
      if (accel_mag > 2.0 || features[6] > 50.0 || features[7] > 60.0 || features[8] > 0.5) {
        handleAnomaly(0.9);  // High confidence anomaly in fallback mode
        activity_detected = true;
      }
    }
    
    last_inference_time = current_time;
  }
  
  // 3. Print status periodically
  if (current_time - last_status_time >= STATUS_INTERVAL_MS) {
    printSystemStatus();
    last_status_time = current_time;
  }
  
  // 4. Handle deep sleep (if enabled)
  #ifdef ENABLE_DEEP_SLEEP
  if (current_time - last_activity_time > DEEP_SLEEP_TIMEOUT_MS) {
    enterDeepSleep();
  }
  #endif
  
  // Small delay to prevent watchdog issues
  if (!activity_detected) {
    delay(1);
  }
}

// ===== Helper Functions =====

/**
 * Enters a critical error state and blinks the LED
 * @param error_msg Error message to print to serial
 */
void enterErrorState(const char* error_msg) {
  #ifdef ENABLE_SERIAL_DEBUG
  Serial.print("CRITICAL ERROR: ");
  Serial.println(error_msg);
  #endif
  
  // Flash LED rapidly to indicate error
  while (true) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
    
    // Allow watchdog to reset the system if we're stuck
    esp_task_wdt_reset();
  }
}

/**
 * Prints system status information to serial
 */
void printSystemStatus() {
  #ifdef ENABLE_SERIAL_DEBUG
  #ifdef ESP32
    static unsigned long last_heap_print = 0;
    unsigned long current_time = millis();
    
    // Only print heap status every 30 seconds to avoid flooding the serial
    if (current_time - last_heap_print > 30000) {
      last_heap_print = current_time;
      
      Serial.println("\n=== System Status ===");
      Serial.print("Uptime: ");
      Serial.print(current_time / 1000);
      Serial.println("s");
      
      Serial.print("Heap: ");
      Serial.print(ESP.getFreeHeap());
      Serial.print(" / ");
      Serial.print(ESP.getHeapSize());
      Serial.println(" bytes free");
      
      Serial.print("Min free heap: ");
      Serial.print(ESP.getMinFreeHeap());
      Serial.println(" bytes");
      
      Serial.print("Anomaly count: ");
      Serial.println(anomaly_count);
      
      Serial.print("Alert active: ");
      Serial.println(alert_triggered ? "YES" : "NO");
      
      // Print TF Lite memory usage if initialized
      if (tflite_initialized) {
        print_tensor_arena_usage();
      }
      
      Serial.println("====================");
    }
  #endif
  #endif
}

/**
 * Puts the device into deep sleep to save power
 */
void enterDeepSleep() {
  #ifdef ENABLE_SERIAL_DEBUG
  Serial.println("Entering deep sleep mode...");
  Serial.flush();
  #endif
  
  // Turn off peripherals
  digitalWrite(LED_PIN, LOW);
  
  // Configure wakeup sources
  #ifdef ESP32
  esp_sleep_enable_timer_wakeup(DEEP_SLEEP_DURATION_US);
  
  // Configure GPIOs to reduce power consumption
  gpio_deep_sleep_hold_dis();
  gpio_hold_dis((gpio_num_t)LED_PIN);
  
  // Enter deep sleep
  esp_deep_sleep_start();
  #endif
}