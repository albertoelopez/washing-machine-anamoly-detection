#ifndef WASHING_MACHINE_UTILS_H
#define WASHING_MACHINE_UTILS_H

#include <SD.h>
#include <SPI.h>
#include "config.h"

// SD Card functions
bool setupSDCard() {
  if (!SD.begin(SD_CS)) {
    Serial.println("SD card initialization failed!");
    return false;
  }
  
  // Check if the file exists, if not, create it with headers
  if (!SD.exists(LOG_FILENAME)) {
    File dataFile = SD.open(LOG_FILENAME, FILE_WRITE);
    if (dataFile) {
      dataFile.println("timestamp,accel_x,accel_y,accel_z,temp,audio_rms,anomaly_score");
      dataFile.close();
      Serial.println("Created new log file");
    } else {
      Serial.println("Error creating log file");
      return false;
    }
  }
  
  Serial.println("SD card initialized");
  return true;
}

void logData(unsigned long timestamp, const float* features, float anomaly_score) {
  File dataFile = SD.open(LOG_FILENAME, FILE_APPEND);
  
  if (dataFile) {
    // Write timestamp
    dataFile.print(timestamp);
    dataFile.print(",");
    
    // Write sensor data
    for (int i = 0; i < 5; i++) {  // First 5 features are sensor readings
      dataFile.print(features[i], 6);
      dataFile.print(",");
    }
    
    // Write anomaly score
    dataFile.println(anomaly_score, 6);
    
    dataFile.close();
  } else {
    Serial.println("Error opening log file");
  }
}

// Helper functions
float calculateMovingAverage(float* scores, int size, float new_score) {
  static int index = 0;
  scores[index] = new_score;
  index = (index + 1) % size;
  
  float sum = 0;
  for (int i = 0; i < size; i++) {
    sum += scores[i];
  }
  
  return sum / size;
}

bool isAnomaly(float score) {
  return score > ANOMALY_THRESHOLD;
}

void triggerAlert() {
  // Implement alert mechanism (e.g., LED, buzzer, etc.)
  digitalWrite(LED_BUILTIN, HIGH);
  delay(1000);
  digitalWrite(LED_BUILTIN, LOW);
  
  // You can also send an alert via serial or other communication
  Serial.println("ALERT: Anomaly detected!");
}

#endif // WASHING_MACHINE_UTILS_H
