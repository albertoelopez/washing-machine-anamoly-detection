#ifndef WASHING_MACHINE_CONFIG_H
#define WASHING_MACHINE_CONFIG_H

// Pin definitions
#define I2C_SDA 21
#define I2C_SCL 22
#define SD_CS 5

// Model configuration
constexpr int NUM_FEATURES = 20;  // Update based on your feature count
constexpr int FEATURE_WINDOW = 10;  // Number of time steps in your model
constexpr int ANOMALY_WINDOW = 5;   // Window size for moving average
constexpr float ANOMALY_THRESHOLD = 0.8;  // Adjust based on your model's behavior

// Sampling configuration
constexpr unsigned long SAMPLE_INTERVAL_MS = 10;  // 100 Hz sampling
constexpr unsigned long INFERENCE_INTERVAL_MS = 1000;  // Run inference every second

// SD Card configuration
constexpr const char* LOG_FILENAME = "/sensor_data.csv";

#endif // WASHING_MACHINE_CONFIG_H
