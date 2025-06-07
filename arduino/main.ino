/*
 * Washing Machine Anomaly Detection
 * 
 * This sketch reads sensor data from an ESP32, runs inference using a TensorFlow Lite model,
 * and detects anomalies in washing machine operation.
 */

#include "src/config.h"
#include "src/sensors.h"
#include "src/tflite_helper.h"
#include "src/utils.h"

// Global variables
float features[NUM_FEATURES] = {0};  // Current sensor readings
float anomaly_scores[ANOMALY_WINDOW] = {0};  // Circular buffer for anomaly scores
int score_index = 0;  // Current position in the circular buffer
bool alert_triggered = false;  // Alert state

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  // Initialize hardware
  setupSensors();
  setupTensorFlow();
  
  // Initialize SD card
  if (!setupSDCard()) {
    Serial.println("SD card initialization failed. Data logging disabled.");
  }
  
  // Set up built-in LED for alerts
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  
  Serial.println("Washing Machine Monitor Ready");
}

void loop() {
  static unsigned long last_sample = 0;
  static unsigned long last_inference = 0;
  static float feature_buffer[FEATURE_WINDOW][NUM_FEATURES] = {0};
  static int buffer_index = 0;
  
  unsigned long current_time = millis();
  
  // 1. Read sensors at fixed interval (100Hz)
  if (current_time - last_sample >= SAMPLE_INTERVAL_MS) {
    last_sample = current_time;
    
    // Read sensor data into the current buffer position
    readSensors(feature_buffer[buffer_index]);
    
    // Move to the next position in the circular buffer
    buffer_index = (buffer_index + 1) % FEATURE_WINDOW;
  }
  
  // 2. Run inference at fixed interval (1Hz)
  if (current_time - last_inference >= INFERENCE_INTERVAL_MS) {
    last_inference = current_time;
    
    // Calculate mean of the feature window
    float mean_features[NUM_FEATURES] = {0};
    for (int i = 0; i < FEATURE_WINDOW; i++) {
      for (int j = 0; j < NUM_FEATURES; j++) {
        mean_features[j] += feature_buffer[i][j];
      }
    }
    
    for (int j = 0; j < NUM_FEATURES; j++) {
      mean_features[j] /= FEATURE_WINDOW;
    }
    
    // Run inference
    float anomaly_score = runInference(mean_features);
    
    // Update moving average of anomaly scores
    float avg_score = calculateMovingAverage(anomaly_scores, ANOMALY_WINDOW, anomaly_score);
    
    // Log data to SD card
    logData(current_time, mean_features, avg_score);
    
    // Check for anomalies
    if (isAnomaly(avg_score) && !alert_triggered) {
      triggerAlert();
      alert_triggered = true;
    } else if (!isAnomaly(avg_score) && alert_triggered) {
      alert_triggered = false;
    }
    
    // Print debug info
    Serial.print("Anomaly score: ");
    Serial.print(anomaly_score, 4);
    Serial.print(" (avg: ");
    Serial.print(avg_score, 4);
    Serial.print(", threshold: ");
    Serial.print(ANOMALY_THRESHOLD, 4);
    Serial.println(")");
  }
  
  // Small delay to prevent watchdog issues
  delay(1);
}