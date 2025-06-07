// washing_machine_monitor.ino
#include "src/config.h"
#include "src/sensors.h"
#include "src/tflite_helper.h"
#include "src/utils.h"

// Global variables
float features[NUM_FEATURES] = {0};
float anomaly_scores[ANOMALY_WINDOW] = {0};
int score_index = 0;
bool alert_triggered = false;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  // Initialize hardware
  setupSensors();
  setupTensorFlow();
  setupSDCard();
  
  Serial.println("Washing Machine Monitor Ready");
}

void loop() {
  static unsigned long lastInference = 0;
  static float feature_buffer[FEATURE_WINDOW][NUM_FEATURES];
  static int buffer_index = 0;
  
  // 1. Read sensors
  readSensors(features);
  
  // 2. Add to feature buffer
  memcpy(feature_buffer[buffer_index], features, sizeof(features));
  buffer_index = (buffer_index + 1) % FEATURE_WINDOW;
  
  // 3. Run inference every second
  if (millis() - lastInference >= 1000) {
    lastInference = millis();
    
    // 4. Prepare input tensor
    float* input_data = tflInterpreter->input(0)->data.f;
    for (int i = 0; i < FEATURE_WINDOW; i++) {
      for (int j = 0; j < NUM_FEATURES; j++) {
        *input_data++ = feature_buffer[(buffer_index + i) % FEATURE_WINDOW][j];
      }
    }
    
    // 5. Run inference
    float anomaly_score = runInference();
    
    // 6. Update moving average
    anomaly_scores[score_index] = anomaly_score;
    score_index = (score_index + 1) % ANOMALY_WINDOW;
    
    float avg_score = 0;
    for (int i = 0; i < ANOMALY_WINDOW; i++) {
      avg_score += anomaly_scores[i];
    }
    avg_score /= ANOMALY_WINDOW;
    
    // 7. Check for anomaly
    if (avg_score > ANOMALY_THRESHOLD && !alert_triggered) {
      alert_triggered = true;
      char alert[100];
      snprintf(alert, sizeof(alert), "ALERT! Anomaly detected. Score: %.4f", avg_score);
      Serial.println(alert);
      logToSD(alert);
      blinkLED(5, 200);
    } else if (avg_score <= ANOMALY_THRESHOLD) {
      alert_triggered = false;
    }
    
    // 8. Log data
    char log_entry[256];
    snprintf(log_entry, sizeof(log_entry), 
             "%lu,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f",
             millis(),
             features[0], features[1], features[2],  // accel x,y,z
             features[3], features[4],               // temp, audio
             anomaly_score, avg_score);
    logToSD(log_entry);
  }
}