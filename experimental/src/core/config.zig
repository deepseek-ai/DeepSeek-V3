const std = @import("std");

/// Global configuration for DeepSeek V3
pub const Config = struct {
    log_level: std.log.Level = .info,
    enable_telemetry: bool = false,
    cache_dir: ?[]const u8 = null,
    
    pub fn loadFromEnv() Config {
        // TODO: Load configuration from environment variables
        return Config{};
    }
}; 