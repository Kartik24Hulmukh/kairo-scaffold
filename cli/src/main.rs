use clap::{Parser, Subcommand};
use kairo_core::client::KairoClient;
use serde_json::json;

#[derive(Parser)]
#[command(name = "kairo")]
#[command(about = "Kairo CLI - Verifiable local document intelligence", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run document extraction
    Run {
        /// Path to the document
        path: String,
        /// Pack to use for extraction (e.g. generic, invoice, paper, contract)
        #[arg(short, long, default_value = "generic")]
        pack: String,
    },
    /// Index a document
    Index {
        /// Path to the document
        path: String,
    },
    /// Ask a question about a document
    Ask {
        /// Document ID or path to the document file
        doc_id: String,
        /// Question to ask
        query: String,
    },
    /// Run diagnostic checks on system health
    Doctor,
    /// Manage API keys in the OS keychain
    Keys {
        #[command(subcommand)]
        subcommand: KeysCommands,
    },
}

#[derive(Subcommand)]
enum KeysCommands {
    /// Set an API key
    Set {
        /// Provider name (openai, anthropic, google)
        provider: String,
        /// API key value
        key: String,
    },
    /// Clear an API key
    Clear {
        /// Provider name (openai, anthropic, google)
        provider: String,
    },
    /// List configured API keys (redacted)
    List,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    // Default local port from SPEC §1
    let client = KairoClient::new("http://127.0.0.1:7438");

    match cli.command {
        Commands::Run { path, pack } => {
            if !std::path::Path::new(&path).exists() {
                eprintln!("Error: file not found {}", path);
                std::process::exit(1);
            }

            match client.index(&path).await {
                Ok((doc_id, _pages, _chunks)) => {
                    match client.extract(&doc_id, &pack).await {
                        Ok(extractions) => {
                            let mut results = serde_json::Map::new();
                            let mut all_blocked = true;

                            for ext in extractions {
                                let method_str = format!("{:?}", ext.method).to_lowercase();
                                if method_str != "block" {
                                    all_blocked = false;

                                    let mut page_val = 0;
                                    let mut bbox_val = serde_json::Value::Null;

                                    if let Some(anchor) = ext.anchors.first() {
                                        page_val = anchor.page;
                                        bbox_val = json!([
                                            anchor.bbox.x0,
                                            anchor.bbox.y0,
                                            anchor.bbox.x1,
                                            anchor.bbox.y1
                                        ]);
                                    }

                                    let entry = json!({
                                        "value": ext.value,
                                        "confidence": ext.confidence,
                                        "page": page_val,
                                        "bbox": bbox_val,
                                        "method": method_str
                                    });
                                    results.insert(ext.field, entry);
                                }
                            }

                            if all_blocked || results.is_empty() {
                                println!("Refused to answer: not verifiable in source text");
                            } else {
                                let json_val = serde_json::Value::Object(results);
                                println!("{}", serde_json::to_string_pretty(&json_val).unwrap());
                            }
                        }
                        Err(e) => {
                            eprintln!("Extraction error: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Indexing error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Index { path } => {
            if !std::path::Path::new(&path).exists() {
                eprintln!("Error: file not found {}", path);
                std::process::exit(1);
            }
            match client.index(&path).await {
                Ok((doc_id, pages, chunks)) => {
                    println!("Successfully indexed. ID: {}, Pages: {}, Chunks: {}", doc_id, pages, chunks);
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Ask { doc_id, query } => {
            let resolved_doc_id = if std::path::Path::new(&doc_id).exists() {
                match client.index(&doc_id).await {
                    Ok((indexed_doc_id, _, _)) => indexed_doc_id,
                    Err(e) => {
                        eprintln!("Error indexing document: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                doc_id
            };

            match client.ask(&resolved_doc_id, &query).await {
                Ok(answer) => {
                    if !answer.grounded || answer.text == "blocked" {
                        println!("Refused to answer: not verifiable in source text");
                    } else {
                        let mut page_val = 0;
                        let mut bbox_val = serde_json::Value::Null;
                        let method_val = "semantic".to_string(); // Default cascade method for ask queries

                        if let Some(citation) = answer.citations.first() {
                            page_val = citation.page;
                            bbox_val = json!([
                                citation.bbox.x0,
                                citation.bbox.y0,
                                citation.bbox.x1,
                                citation.bbox.y1
                            ]);
                        }

                        let result = json!({
                            "answer": {
                                "value": answer.text,
                                "confidence": 1.0,
                                "page": page_val,
                                "bbox": bbox_val,
                                "method": method_val
                            }
                        });
                        println!("{}", serde_json::to_string_pretty(&result).unwrap());
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Doctor => {
            match kairo_core::doctor::run_diagnostics("http://127.0.0.1:7438").await {
                Ok((sidecar, sqlite, vector, mode, license)) => {
                    println!("+-------------------------------------------------+--------+");
                    println!("| Check Description                               | Status |");
                    println!("+-------------------------------------------------+--------+");
                    println!("| Sidecar Reachable (http://127.0.0.1:7438/docs)  | {:<6} |", sidecar);
                    println!("| SQLite Database Writable (.kairo/kairo.db)      | {:<6} |", sqlite);
                    println!("| Vector Store Writable (LanceDB / Qdrant)        | {:<6} |", vector);
                    println!("| CPU/GPU Mode Detected                           | {:<6} |", mode);
                    println!("| CI License Compliance Check                     | {:<6} |", license);
                    println!("+-------------------------------------------------+--------+");
                    
                    if sidecar == "FAIL" || sqlite == "FAIL" || vector == "FAIL" || license == "FAIL" {
                        eprintln!("\nError: Critical diagnostics failed.");
                        std::process::exit(1);
                    } else {
                        println!("\nAll diagnostics green!");
                        std::process::exit(0);
                    }
                }
                Err(e) => {
                    eprintln!("Doctor failed to run diagnostics: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Keys { subcommand } => {
            let python_path = if std::path::Path::new("kernel/sidecar/.venv/Scripts/python.exe").exists() {
                "kernel/sidecar/.venv/Scripts/python.exe"
            } else if std::path::Path::new("kernel/sidecar/.venv/bin/python").exists() {
                "kernel/sidecar/.venv/bin/python"
            } else {
                "python"
            };

            let mut cmd = std::process::Command::new(python_path);
            cmd.arg("-m").arg("cli").arg("keys");
            match subcommand {
                KeysCommands::Set { provider, key } => {
                    cmd.arg("set").arg(provider).arg(key);
                }
                KeysCommands::Clear { provider } => {
                    cmd.arg("clear").arg(provider);
                }
                KeysCommands::List => {
                    cmd.arg("list");
                }
            }

            match cmd.output() {
                Ok(output) => {
                    if output.status.success() {
                        print!("{}", String::from_utf8_lossy(&output.stdout));
                    } else {
                        eprint!("{}", String::from_utf8_lossy(&output.stderr));
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Error executing keys command via Python: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}
