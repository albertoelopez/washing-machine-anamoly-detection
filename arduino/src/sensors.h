#ifndef WASHING_MACHINE_SENSORS_H
#define WASHING_MACHINE_SENSORS_H

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_MLX90614.h>
#include "config.h"

// ===== Sensor Configuration =====
// Sensor objects with error tracking
typedef struct {
  Adafruit_MPU6050 mpu;
  Adafruit_MLX90614 mlx;
  bool mpu_initialized = false;
  bool mlx_initialized = false;
  bool audio_initialized = false;
  uint8_t consecutive_errors = 0;
} SensorState;

// Global sensor state
static SensorState sensor_state;

// Audio input configuration
const int AUDIO_PIN = 34;  // Analog pin for microphone
const int AUDIO_SAMPLE_WINDOW = 50;  // ms of sample to take for RMS calculation
const int AUDIO_SAMPLE_RATE = 10000;  // Hz - 10kHz sampling for audio

// ===== Function Declarations =====
bool setupSensors();
bool readSensors(float* features);
void sleepSensors();
void wakeSensors();
float readAudioRMS(uint8_t pin, uint32_t sample_window_ms = AUDIO_SAMPLE_WINDOW);

/**
 * Initialize all sensors
 * @return true if all sensors initialized successfully, false otherwise
 */
bool setupSensors() {
  bool success = true;
  
  // Initialize I2C with timeout and clock stretching
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);  // 400kHz I2C clock
  
  #ifdef ENABLE_SERIAL_DEBUG
  Serial.println("Initializing sensors...");
  #endif
  
  // Initialize MPU6050 (accelerometer)
  #ifdef ENABLE_SERIAL_DEBUG
  Serial.print("Initializing MPU6050... ");
  #endif
  
  sensor_state.mpu_initialized = sensor_state.mpu.begin(0x68, &Wire);
  
  if (sensor_state.mpu_initialized) {
    #ifdef ENABLE_SERIAL_DEBUG
    Serial.println("OK");
    #endif
    
    // Configure MPU6050
    sensor_state.mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    sensor_state.mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    sensor_state.mpu.setHighPassFilter(MPU6050_HIGHPASS_0_63_HZ);
  } else {
    #ifdef ENABLE_SERIAL_DEBUG
    Serial.println("FAILED");
    #endif
    success = false;
  }
  
  // Initialize MLX90614 (temperature)
  #ifdef ENABLE_SERIAL_DEBUG
  Serial.print("Initializing MLX90614... ");
  #endif
  
  sensor_state.mlx_initialized = sensor_state.mlx.begin();
  
  #ifdef ENABLE_SERIAL_DEBUG
  if (sensor_state.mlx_initialized) {
    Serial.println("OK");
  } else {
    Serial.println("FAILED (check wiring)");
  }
  #endif
  
  // Configure audio input
  pinMode(AUDIO_PIN, INPUT);
  sensor_state.audio_initialized = true;
  
  #ifdef ENABLE_SERIAL_DEBUG
  if (success) {
    Serial.println("All sensors initialized successfully");
  } else {
    Serial.println("Warning: Some sensors failed to initialize");
  }
  #endif
  
  return success;
}

/**
 * Read all sensors and populate the features array
 * @param features Array to store the sensor readings
 * @return true if successful, false if there was an error
 */
bool readSensors(float* features) {
  if (!features) {
    return false;
  }
  
  bool success = true;
  sensors_event_t a, g, temp;
  
  // Read accelerometer if initialized
  if (sensor_state.mpu_initialized) {
    if (!sensor_state.mpu.getEvent(&a, &g, &temp)) {
      sensor_state.consecutive_errors++;
      success = false;
      #ifdef ENABLE_SERIAL_DEBUG
      Serial.println("Error reading MPU6050");
      #endif
    } else {
      sensor_state.consecutive_errors = 0;  // Reset on successful read
    }
  } else {
    // Use zeros if sensor is not available
    a.acceleration.x = a.acceleration.y = a.acceleration.z = 0;
    g.gyro.x = g.gyro.y = g.gyro.z = 0;
    temp.temperature = 0;
  }
  
  // Read temperature if MLX90614 is initialized
  float ambient_temp = 0.0f, object_temp = 0.0f;
  if (sensor_state.mlx_initialized) {
    ambient_temp = sensor_state.mlx.readAmbientTempC();
    object_temp = sensor_state.mlx.readObjectTempC();
    
    // Check for reasonable temperature values
    if (isnan(ambient_temp) || isnan(object_temp) || 
        ambient_temp < -40.0f || ambient_temp > 85.0f ||
        object_temp < -40.0f || object_temp > 300.0f) {
      success = false;
      #ifdef ENABLE_SERIAL_DEBUG
      Serial.println("Error: Invalid temperature reading");
      #endif
    }
  }
  
  // Read audio RMS if audio is initialized
  float audio_rms = 0.0f;
  if (sensor_state.audio_initialized) {
    audio_rms = readAudioRMS(AUDIO_PIN);
    
    // Check for reasonable audio value
    if (isnan(audio_rms) || audio_rms < 0 || audio_rms > 1024) {
      success = false;
      #ifdef ENABLE_SERIAL_DEBUG
      Serial.println("Error: Invalid audio reading");
      #endif
    }
  }
  
  // Populate features array (adjust indices based on your model)
  features[0] = a.acceleration.x;
  features[1] = a.acceleration.y;
  features[2] = a.acceleration.z;
  features[3] = g.gyro.x;
  features[4] = g.gyro.y;
  features[5] = g.gyro.z;
  features[6] = ambient_temp;
  features[7] = object_temp;
  features[8] = audio_rms;
  
  // Add more features as needed
  for (int i = 9; i < NUM_FEATURES; i++) {
    features[i] = 0.0f;  // Fill remaining features with zeros
  }
  
  return success && (sensor_state.consecutive_errors < MAX_CONSECUTIVE_ERRORS);
}

/**
 * Put sensors into low-power mode
 */
void sleepSensors() {
  #ifdef ENABLE_SERIAL_DEBUG
  Serial.println("Putting sensors to sleep...");
  #endif
  
  if (sensor_state.mpu_initialized) {
    sensor_state.mpu.enableSleep(true);
  }
  
  // MLX90614 doesn't have a sleep mode, but we can reduce the update rate
  // or turn it off completely if needed
  
  // Disable ADC for audio to save power
  if (sensor_state.audio_initialized) {
    // ESP32 specific: disable ADC
    #ifdef ESP32
    adc_power_release();
    #endif
  }
}

/**
 * Wake up sensors from low-power mode
 */
void wakeSensors() {
  #ifdef ENABLE_SERIAL_DEBUG
  Serial.println("Waking up sensors...");
  #endif
  
  if (sensor_state.mpu_initialized) {
    sensor_state.mpu.enableSleep(false);
    // Re-initialize MPU6050 after sleep
    setupSensors();
  }
  
  // Re-enable ADC for audio
  if (sensor_state.audio_initialized) {
    #ifdef ESP32
    adc_power_acquire();
    #endif
  }
}

/**
 * Read RMS value from audio input
 * @param pin Analog pin number
 * @param sample_window_ms Sampling window in milliseconds
 * @return RMS value of the audio signal
 */
float readAudioRMS(uint8_t pin, uint32_t sample_window_ms) {
  uint32_t start_time = millis();
  uint32_t peak_to_peak = 0;
  uint16_t signal_max = 0;
  uint16_t signal_min = 1024;
  uint32_t samples = 0;
  uint32_t sum_squares = 0;
  
  // Sample for the specified window
  while ((millis() - start_time) < sample_window_ms) {
    uint16_t sample = analogRead(pin);
    
    // Track min/max for peak-to-peak
    if (sample < signal_min) signal_min = sample;
    if (sample > signal_max) signal_max = sample;
    
    // Calculate sum of squares for RMS
    sum_squares += (uint32_t)sample * sample;
    samples++;
    
    // Small delay to achieve desired sample rate
    delayMicroseconds(1000000 / AUDIO_SAMPLE_RATE);
  }
  
  // Calculate RMS (Root Mean Square)
  float rms = sqrt(sum_squares / (float)samples);
  
  // Calculate peak-to-peak amplitude
  peak_to_peak = signal_max - signal_min;
  
  // Return normalized value (0-1.0)
  return rms / 1024.0f;
}

#endif // WASHING_MACHINE_SENSORS_H
