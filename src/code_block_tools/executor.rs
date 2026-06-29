//! Tool execution engine for running external formatters and linters.
//!
//! This module handles the actual execution of external tools via stdin/stdout,
//! with timeout support and lazy tool availability checking.

use super::config::ToolDefinition;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
#[cfg(not(target_os = "wasi"))]
use std::io::{Read, Write};
#[cfg(not(target_os = "wasi"))]
use std::process::{Command, Stdio};
#[cfg(not(target_os = "wasi"))]
use std::thread;
#[cfg(not(target_os = "wasi"))]
use std::time::{Duration, Instant};

// Because WASI does not include process spawning, we define a custom ABI for it.
#[cfg(target_os = "wasi")]
mod host {
    #[link(wasm_import_module = "rumdl")]
    unsafe extern "C" {
        /// Returns non-zero if a tool with the given name is available.
        pub fn check_tool_exists(name_ptr: *const u8, name_len: usize) -> i32;

        /// Executes a tool, passing `args` (each framed as a u32 LE length
        /// followed by that many bytes) and `stdin`.
        ///
        /// The host writes the captured stdout/stderr into guest memory it
        /// allocates via [`rumdl_wasm_alloc`], storing the pointer and length
        /// at the provided out-params. Returns the tool's exit code.
        #[allow(clippy::too_many_arguments)]
        pub fn execute_tool(
            name_ptr: *const u8,
            name_len: usize,
            args_ptr: *const u8,
            args_len: usize,
            stdin_ptr: *const u8,
            stdin_len: usize,
            timeout_ms: u64,
            out_stdout_ptr: *mut usize,
            out_stdout_len: *mut usize,
            out_stderr_ptr: *mut usize,
            out_stderr_len: *mut usize,
        ) -> i32;
    }
}

/// Allocates `len` bytes in guest linear memory for the host to fill.
///
/// Called by the WASI host (not from Rust) to return tool output. The matching
/// free happens in the guest in [`read_host_string`] after the bytes have been copied out.
#[cfg(target_os = "wasi")]
#[unsafe(no_mangle)]
pub extern "C" fn rumdl_wasm_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::NonNull::<u8>::dangling().as_ptr();
    }
    // SAFETY: len > 0, align 1 is always valid for u8.
    unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(len, 1)) }
}

/// Copies a host-allocated `[ptr, len)` byte range into an owned String and
/// frees the allocation made by [`rumdl_wasm_alloc`].
#[cfg(target_os = "wasi")]
fn read_host_string(ptr: usize, len: usize) -> String {
    if ptr == 0 || len == 0 {
        return String::new();
    }
    // SAFETY: the host allocated [ptr, len) via rumdl_wasm_alloc (align 1).
    unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const u8, len);
        let s = String::from_utf8_lossy(slice).into_owned();
        std::alloc::dealloc(ptr as *mut u8, std::alloc::Layout::from_size_align_unchecked(len, 1));
        s
    }
}

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// Standard output from the tool.
    pub stdout: String,
    /// Standard error from the tool.
    pub stderr: String,
    /// Exit code (0 typically means success).
    pub exit_code: i32,
    /// Whether the tool executed successfully (exit code 0).
    pub success: bool,
}

/// Error during tool execution.
#[derive(Debug, Clone)]
pub enum ExecutorError {
    /// Tool binary not found in PATH.
    ToolNotFound { tool: String },
    /// Tool execution failed.
    ExecutionFailed { tool: String, message: String },
    /// Tool execution timed out.
    Timeout { tool: String, timeout_ms: u64 },
    /// I/O error during execution.
    IoError { message: String },
}

impl std::fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ToolNotFound { tool } => {
                write!(f, "Tool '{tool}' not found in PATH")
            }
            Self::ExecutionFailed { tool, message } => {
                write!(f, "Tool '{tool}' failed: {message}")
            }
            Self::Timeout { tool, timeout_ms } => {
                write!(f, "Tool '{tool}' timed out after {timeout_ms}ms")
            }
            Self::IoError { message } => {
                write!(f, "I/O error: {message}")
            }
        }
    }
}

impl std::error::Error for ExecutorError {}

/// Executor for running external tools.
///
/// Caches tool availability checks for efficiency.
pub struct ToolExecutor {
    /// Cache of tool availability checks (tool name -> available).
    tool_cache: Arc<Mutex<HashMap<String, bool>>>,
    /// Default timeout in milliseconds.
    default_timeout_ms: u64,
}

impl ToolExecutor {
    /// Create a new executor with the given default timeout.
    pub fn new(default_timeout_ms: u64) -> Self {
        Self {
            tool_cache: Arc::new(Mutex::new(HashMap::new())),
            default_timeout_ms,
        }
    }

    /// Check if a tool is available (lazy, cached).
    pub fn is_tool_available(&self, tool_name: &str) -> bool {
        // Check cache first
        {
            let cache = self.tool_cache.lock().unwrap();
            if let Some(&available) = cache.get(tool_name) {
                return available;
            }
        }

        // Check if tool exists using `which` on Unix or `where` on Windows
        let available = self.check_tool_exists(tool_name);

        // Cache the result
        {
            let mut cache = self.tool_cache.lock().unwrap();
            cache.insert(tool_name.to_string(), available);
        }

        available
    }

    /// Check if a tool binary exists.
    fn check_tool_exists(&self, tool_name: &str) -> bool {
        #[cfg(unix)]
        {
            Command::new("which")
                .arg(tool_name)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        }

        #[cfg(windows)]
        {
            Command::new("where")
                .arg(tool_name)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        }

        #[cfg(target_os = "wasi")]
        {
            !tool_name.is_empty() && unsafe { host::check_tool_exists(tool_name.as_ptr(), tool_name.len()) != 0 }
        }

        #[cfg(not(any(unix, windows, target_os = "wasi")))]
        {
            // Other WASM and platforms without process support: tools unavailable
            let _ = tool_name;
            false
        }
    }

    /// Execute a tool with the given input by spawning a child process.
    ///
    /// # Arguments
    /// * `tool_def` - Tool definition with command and arguments
    /// * `input` - Content to pass via stdin
    /// * `is_format_mode` - Whether to use format_args (true) or lint_args (false)
    /// * `timeout_ms` - Optional timeout override
    ///
    /// # Returns
    /// Tool output on success, or an error.
    #[cfg(not(target_os = "wasi"))]
    pub fn execute(
        &self,
        tool_def: &ToolDefinition,
        input: &str,
        is_format_mode: bool,
        timeout_ms: Option<u64>,
    ) -> Result<ToolOutput, ExecutorError> {
        if tool_def.command.is_empty() {
            return Err(ExecutorError::ExecutionFailed {
                tool: "unknown".to_string(),
                message: "Empty command".to_string(),
            });
        }

        let tool_name = &tool_def.command[0];

        // Check tool availability (lazy, cached)
        if !self.is_tool_available(tool_name) {
            return Err(ExecutorError::ToolNotFound {
                tool: tool_name.clone(),
            });
        }

        // Build command
        let mut cmd = Command::new(tool_name);

        // Add base arguments
        if tool_def.command.len() > 1 {
            cmd.args(&tool_def.command[1..]);
        }

        // Add mode-specific arguments
        let extra_args = if is_format_mode {
            &tool_def.format_args
        } else {
            &tool_def.lint_args
        };
        if !extra_args.is_empty() {
            cmd.args(extra_args);
        }

        // Configure stdin/stdout
        if tool_def.stdin {
            cmd.stdin(Stdio::piped());
        }
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Spawn process
        let mut child = cmd.spawn().map_err(|e| ExecutorError::IoError {
            message: format!("Failed to spawn '{tool_name}': {e}"),
        })?;

        let mut stdout_handle = child
            .stdout
            .take()
            .map(|stdout| thread::spawn(move || read_pipe_to_string(stdout)));
        let mut stderr_handle = child
            .stderr
            .take()
            .map(|stderr| thread::spawn(move || read_pipe_to_string(stderr)));

        // Write stdin if required.
        // BrokenPipe is ignored: the tool may exit before consuming all input
        // (e.g., `true` or a linter that validates without reading fully).
        if tool_def.stdin
            && let Some(mut stdin) = child.stdin.take()
            && let Err(e) = stdin.write_all(input.as_bytes())
            && e.kind() != std::io::ErrorKind::BrokenPipe
        {
            return Err(ExecutorError::IoError {
                message: format!("Failed to write to stdin: {e}"),
            });
        }

        // Wait for completion with timeout
        let timeout = Duration::from_millis(timeout_ms.unwrap_or(self.default_timeout_ms));
        let status = if timeout.is_zero() {
            child.wait().map_err(|e| ExecutorError::IoError {
                message: format!("Failed to wait for '{tool_name}': {e}"),
            })?
        } else {
            let start = Instant::now();
            loop {
                if let Some(status) = child.try_wait().map_err(|e| ExecutorError::IoError {
                    message: format!("Failed to poll '{tool_name}': {e}"),
                })? {
                    break status;
                }
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = join_reader(stdout_handle.take());
                    let _ = join_reader(stderr_handle.take());
                    return Err(ExecutorError::Timeout {
                        tool: tool_name.clone(),
                        timeout_ms: timeout.as_millis() as u64,
                    });
                }
                thread::sleep(Duration::from_millis(10));
            }
        };

        let stdout = join_reader(stdout_handle.take()).map_err(|e| ExecutorError::IoError { message: e })?;
        let stderr = join_reader(stderr_handle.take()).map_err(|e| ExecutorError::IoError { message: e })?;
        let exit_code = status.code().unwrap_or(-1);

        Ok(ToolOutput {
            stdout,
            stderr,
            exit_code,
            success: status.success(),
        })
    }

    /// Execute a tool with the given input by delegating to the WASI host.
    ///
    /// # Arguments
    /// * `tool_def` - Tool definition with command and arguments
    /// * `input` - Content to pass via stdin
    /// * `is_format_mode` - Whether to use format_args (true) or lint_args (false)
    /// * `timeout_ms` - Optional timeout override
    ///
    /// # Returns
    /// Tool output on success, or an error.
    #[cfg(target_os = "wasi")]
    pub fn execute(
        &self,
        tool_def: &ToolDefinition,
        input: &str,
        is_format_mode: bool,
        timeout_ms: Option<u64>,
    ) -> Result<ToolOutput, ExecutorError> {
        if tool_def.command.is_empty() {
            return Err(ExecutorError::ExecutionFailed {
                tool: "unknown".to_string(),
                message: "Empty command".to_string(),
            });
        }

        let tool_name = &tool_def.command[0];

        // Check tool availability (lazy, cached)
        if !self.is_tool_available(tool_name) {
            return Err(ExecutorError::ToolNotFound {
                tool: tool_name.clone(),
            });
        }

        let extra_args = if is_format_mode {
            &tool_def.format_args
        } else {
            &tool_def.lint_args
        };
        let stdin_input = if tool_def.stdin { input } else { "" };
        let timeout_ms = timeout_ms.unwrap_or(self.default_timeout_ms);

        // Serialize arguments length-prefixed (u32 LE length + bytes per arg)
        // so that args containing any byte (including NUL) are unambiguous.
        let mut args: Vec<u8> = Vec::new();
        for arg in tool_def.command[1..].iter().chain(extra_args.iter()) {
            args.extend_from_slice(&(arg.len() as u32).to_le_bytes());
            args.extend_from_slice(arg.as_bytes());
        }

        let mut stdout_ptr: usize = 0;
        let mut stdout_len: usize = 0;
        let mut stderr_ptr: usize = 0;
        let mut stderr_len: usize = 0;

        // SAFETY: pointers/lengths reference live buffers for the call's
        // duration; the host writes output pointers into the out-params.
        let exit_code = unsafe {
            host::execute_tool(
                tool_name.as_ptr(),
                tool_name.len(),
                args.as_ptr(),
                args.len(),
                stdin_input.as_ptr(),
                stdin_input.len(),
                timeout_ms,
                &mut stdout_ptr,
                &mut stdout_len,
                &mut stderr_ptr,
                &mut stderr_len,
            )
        };

        Ok(ToolOutput {
            stdout: read_host_string(stdout_ptr, stdout_len),
            stderr: read_host_string(stderr_ptr, stderr_len),
            exit_code,
            success: exit_code == 0,
        })
    }

    /// Execute a tool for formatting (returns formatted content).
    pub fn format(
        &self,
        tool_def: &ToolDefinition,
        input: &str,
        timeout_ms: Option<u64>,
    ) -> Result<String, ExecutorError> {
        let output = self.execute(tool_def, input, true, timeout_ms)?;

        if output.success && tool_def.stdout {
            Ok(output.stdout)
        } else if !output.success {
            let exit_code = output.exit_code;
            let stderr = &output.stderr;
            Err(ExecutorError::ExecutionFailed {
                tool: tool_def.command.first().cloned().unwrap_or_default(),
                message: format!("Exit code {exit_code}: {stderr}"),
            })
        } else {
            // Tool doesn't output to stdout, which is unusual for a formatter
            Err(ExecutorError::ExecutionFailed {
                tool: tool_def.command.first().cloned().unwrap_or_default(),
                message: "Formatter doesn't output to stdout".to_string(),
            })
        }
    }

    /// Execute a tool for linting (returns diagnostics).
    pub fn lint(
        &self,
        tool_def: &ToolDefinition,
        input: &str,
        timeout_ms: Option<u64>,
    ) -> Result<ToolOutput, ExecutorError> {
        self.execute(tool_def, input, false, timeout_ms)
    }
}

#[cfg(not(target_os = "wasi"))]
fn read_pipe_to_string<R: Read>(mut pipe: R) -> std::io::Result<String> {
    let mut buf = Vec::new();
    pipe.read_to_end(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

#[cfg(not(target_os = "wasi"))]
fn join_reader(handle: Option<thread::JoinHandle<std::io::Result<String>>>) -> Result<String, String> {
    match handle {
        Some(handle) => match handle.join() {
            Ok(res) => res.map_err(|e| format!("Failed to read output: {e}")),
            Err(_) => Err("Output reader thread panicked".to_string()),
        },
        None => Ok(String::new()),
    }
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::new(30_000) // 30 seconds default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let executor = ToolExecutor::new(10_000);
        // Just verify it creates successfully
        assert_eq!(executor.default_timeout_ms, 10_000);
    }

    #[test]
    fn test_tool_not_found() {
        let executor = ToolExecutor::default();
        let tool_def = ToolDefinition {
            command: vec!["nonexistent-tool-xyz123".to_string()],
            stdin: true,
            stdout: true,
            lint_args: vec![],
            format_args: vec![],
        };

        let result = executor.execute(&tool_def, "test", false, None);
        assert!(matches!(result, Err(ExecutorError::ToolNotFound { .. })));
    }

    #[test]
    fn test_empty_command() {
        let executor = ToolExecutor::default();
        let tool_def = ToolDefinition {
            command: vec![],
            stdin: true,
            stdout: true,
            lint_args: vec![],
            format_args: vec![],
        };

        let result = executor.execute(&tool_def, "test", false, None);
        assert!(matches!(result, Err(ExecutorError::ExecutionFailed { .. })));
    }

    // Integration tests with real tools would go here, but are skipped
    // in unit tests since they require the tools to be installed.

    #[test]
    #[ignore = "requires 'cat' to be available"]
    fn test_execute_cat() {
        let executor = ToolExecutor::default();
        let tool_def = ToolDefinition {
            command: vec!["cat".to_string()],
            stdin: true,
            stdout: true,
            lint_args: vec![],
            format_args: vec![],
        };

        let result = executor.execute(&tool_def, "hello world", false, None);
        let output = result.expect("cat should succeed");
        assert!(output.success);
        assert_eq!(output.stdout.trim(), "hello world");
    }

    #[test]
    #[cfg(unix)]
    #[ignore = "requires 'sleep' to be available"]
    fn test_timeout() {
        let executor = ToolExecutor::new(5);
        let tool_def = ToolDefinition {
            command: vec!["sleep".to_string(), "1".to_string()],
            stdin: false,
            stdout: true,
            lint_args: vec![],
            format_args: vec![],
        };

        let result = executor.execute(&tool_def, "", false, Some(5));
        assert!(matches!(result, Err(ExecutorError::Timeout { .. })));
    }
}
