use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandChild;
use std::time::{Duration, Instant};

pub struct SidecarState {
    pub child: Mutex<Option<CommandChild>>,
}

fn check_health_tcp(timeout: Duration) -> bool {
    let start = Instant::now();
    // Parse target address: local loopback on port 7438
    let addr = match "127.0.0.1:7438".parse::<std::net::SocketAddr>() {
        Ok(a) => a,
        Err(_) => return false,
    };

    while start.elapsed() < timeout {
        if let Ok(mut stream) = std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(150)) {
            use std::io::{Write, Read};
            let request = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
            if stream.write_all(request.as_bytes()).is_ok() {
                let mut buffer = [0; 256];
                if stream.read(&mut buffer).is_ok() {
                    let response = String::from_utf8_lossy(&buffer);
                    if response.contains("HTTP/1.1 200") || response.contains("HTTP/1.0 200") {
                        return true;
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

pub fn start_sidecar(app: &tauri::App) -> Result<(), String> {
    let shell = app.shell();
    
    // Spawn sidecar: "kairo-sidecar" is the target bin in tauri.conf.json
    println!("==> Spawning managed sidecar process: kairo-sidecar");
    let sidecar_command = shell
        .sidecar("kairo-sidecar")
        .map_err(|e| format!("Failed to configure sidecar: {}", e))?;
        
    let (_rx, child) = sidecar_command
        .spawn()
        .map_err(|e| format!("Failed to spawn sidecar: {}", e))?;
        
    // Save state
    app.manage(SidecarState {
        child: Mutex::new(Some(child)),
    });
    
    // Health-check block (budget: <2s)
    let ready = check_health_tcp(Duration::from_millis(2000));
    if !ready {
        return Err("Sidecar failed to start or respond to health check in time".to_string());
    }
    println!("==> Sidecar spawned and verified healthy.");
    
    Ok(())
}

pub fn stop_sidecar(app_handle: &tauri::AppHandle) {
    if let Some(state) = app_handle.try_state::<SidecarState>() {
        let mut guard = state.child.lock().unwrap();
        if let Some(child) = guard.take() {
            println!("==> Terminating managed sidecar process");
            let _ = child.kill();
        }
    }
}
