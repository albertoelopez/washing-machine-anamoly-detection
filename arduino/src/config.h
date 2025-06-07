#ifndef WASHING_MACHINE_CONFIG_H
#define WASHING_MACHINE_CONFIG_H

// ===== Pin Definitions =====
#define I2C_SDA 21
#define I2C_SCL 22
#define SD_CS 5
#define LED_PIN LED_BUILTIN  // Built-in LED for status indication

// ===== Model Configuration =====
constexpr int NUM_FEATURES = 20;     // Number of input features for the model
constexpr int FEATURE_WINDOW = 10;    // Number of time steps in the sliding window
constexpr int ANOMALY_WINDOW = 5;     // Window size for moving average of anomaly scores
constexpr float ANOMALY_THRESHOLD = 0.8f;  // Threshold for anomaly detection (0.0-1.0)

// ===== Timing Configuration =====
constexpr unsigned long SAMPLE_INTERVAL_MS = 10;      // 100 Hz sampling rate
constexpr unsigned long INFERENCE_INTERVAL_MS = 1000;  // Run inference every second
constexpr unsigned long STATUS_INTERVAL_MS = 5000;     // Print status every 5 seconds

// ===== Power Management =====
// Uncomment to enable deep sleep between inferences
// #define ENABLE_DEEP_SLEEP
constexpr uint64_t DEEP_SLEEP_DURATION_US = 10 * 1000000;  // 10 seconds

// ===== Error Handling =====
constexpr int MAX_CONSECUTIVE_ERRORS = 5;  // Maximum sensor read errors before reset

// ===== SD Card Configuration =====
constexpr const char* LOG_FILENAME = "/sensor_data.csv";
constexpr const char* ANOMALY_LOG_FILENAME = "/anomalies.csv";

// ===== Debug Options =====
#define ENABLE_SERIAL_DEBUG   // Comment out to disable debug output
#define ENABLE_TFLITE_PROFILING  // Enable TensorFlow Lite profiling

// ===== Memory Configuration =====
// TensorFlow Lite Micro memory configuration
#if defined(ESP32) || defined(ESP8266)
constexpr int TENSOR_ARENA_SIZE = 12 * 1024;  // 12KB for ESP32
#else
constexpr int TENSOR_ARENA_SIZE = 8 * 1024;    // 8KB for other platforms
#endif

// ===== Watchdog Configuration =====
constexpr int WDT_TIMEOUT_S = 30;  // Watchdog timeout in seconds

// ===== Hardware-Specific Configuration =====
#ifdef ESP32
// ESP32-specific settings
#define USE_RTC_MEMORY        // Use RTC memory for persistence across deep sleep
#define ENABLE_HW_WATCHDOG    // Enable hardware watchdog
#endif

#endif // WASHING_MACHINE_CONFIG_H
