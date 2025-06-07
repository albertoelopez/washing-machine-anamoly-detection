#ifndef WASHING_MACHINE_SENSORS_H
#define WASHING_MACHINE_SENSORS_H

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_MLX90614.h>
#include "config.h"

// Sensor objects
Adafruit_MPU6050 mpu;
Adafruit_MLX90614 mlx = Adafruit_MLX90614();

// Audio input
const int AUDIO_PIN = 34;  // Analog pin for microphone

void setupSensors() {
  // Initialize I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  
  // Initialize MPU6050 (accelerometer)
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  
  // Configure MPU6050
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  // Initialize MLX90614 (temperature)
  if (!mlx.begin()) {
    Serial.println("Failed to find MLX90614 sensor");
    while (1) {
      delay(10);
    }
  }
  
  // Configure audio input
  pinMode(AUDIO_PIN, INPUT);
  
  Serial.println("Sensors initialized");
}

void readSensors(float* features) {
  // Read accelerometer
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  
  // Read temperature
  float ambient_temp = mlx.readAmbientTempC();
  float object_temp = mlx.readObjectTempC();
  
  // Read audio (RMS calculation)
  unsigned long start_time = millis();
  float sum = 0;
  int samples = 0;
  
  while (millis() - start_time < 10) {  // Sample for 10ms
    int sample = analogRead(AUDIO_PIN) - 2048;  // Center around 0
    sum += sample * sample;
    samples++;
  }
  
  float rms = samples > 0 ? sqrt(sum / samples) : 0;
  
  // Update features (example - adjust based on your feature extraction)
  features[0] = a.acceleration.x;
  features[1] = a.acceleration.y;
  features[2] = a.acceleration.z;
  features[3] = ambient_temp;
  features[4] = object_temp;
  features[5] = rms;
  // Add more features as needed
}

#endif // WASHING_MACHINE_SENSORS_H
