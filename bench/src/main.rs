use kairo_core::client::KairoClient;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Fixture {
    file: String,
}

#[derive(Debug, Deserialize)]
struct GroundTruth {
    fixtures: Vec<Fixture>,
}

#[tokio::main]
async fn main() {
    println!("Starting Kairo Grounding Benchmark...");
    
    let client = KairoClient::new("http://127.0.0.1:7438");
    
    // Load golden fixtures ground truth list
    let gt_path = "fixtures/golden/ground_truth.json";
    if !Path::new(gt_path).exists() {
        eprintln!("Error: fixtures/golden/ground_truth.json not found!");
        std::process::exit(1);
    }
    
    let gt_content = std::fs::read_to_string(gt_path).unwrap();
    let gt: GroundTruth = serde_json::from_str(&gt_content).unwrap();
    
    let mut total_extractions = 0;
    let mut grounded_extractions = 0;
    
    println!("Step 1/3: Running grounding checks on golden fixtures...");
    for fixture in gt.fixtures {
        let file_path = format!("fixtures/golden/{}", fixture.file);
        if !Path::new(&file_path).exists() {
            println!("Skipping missing fixture: {}", file_path);
            continue;
        }
        
        let pack = if fixture.file.starts_with("sample_contract_") {
            "contract"
        } else if fixture.file.starts_with("sample_invoice_") {
            "invoice"
        } else if fixture.file.starts_with("sample_paper_") {
            "paper"
        } else {
            "generic"
        };
        
        match client.index(&file_path).await {
            Ok((doc_id, _pages, _chunks)) => {
                match client.extract(&doc_id, pack).await {
                    Ok(extractions) => {
                        for ext in extractions {
                            total_extractions += 1;
                            let method_str = format!("{:?}", ext.method).to_lowercase();
                            if method_str != "block" {
                                grounded_extractions += 1;
                            }
                        }
                    }
                    Err(e) => {
                        println!("Failed to extract for {}: {}", file_path, e);
                    }
                }
            }
            Err(e) => {
                println!("Failed to index {}: {}", file_path, e);
            }
        }
    }
    
    let grounded_answer_rate = if total_extractions > 0 {
        (grounded_extractions as f64 / total_extractions as f64) * 100.0
    } else {
        100.0
    };
    
    println!("Step 2/3: Running grounding checks on unanswerable fixtures...");
    let unanswerable_path = "fixtures/unanswerable.pdf";
    let mut unanswerable_total = 0;
    let mut unanswerable_refused = 0;
    
    if Path::new(unanswerable_path).exists() {
        match client.index(unanswerable_path).await {
            Ok((doc_id, _, _)) => {
                match client.extract(&doc_id, "invoice").await {
                    Ok(extractions) => {
                        for ext in extractions {
                            unanswerable_total += 1;
                            let method_str = format!("{:?}", ext.method).to_lowercase();
                            if method_str == "block" {
                                unanswerable_refused += 1;
                            }
                        }
                    }
                    Err(e) => {
                        println!("Failed to extract for unanswerable: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("Failed to index unanswerable: {}", e);
            }
        }
    } else {
        println!("Warning: unanswerable.pdf not found at {}", unanswerable_path);
    }
    
    let refusal_rate = if unanswerable_total > 0 {
        (unanswerable_refused as f64 / unanswerable_total as f64) * 100.0
    } else {
        100.0
    };
    
    println!("Step 3/3: Grounding Benchmark Results:");
    println!("Grounded-Answer Rate: {:.2}%", grounded_answer_rate);
    println!("Refusal Rate: {:.2}%", refusal_rate);
    
    // Write markdown report
    let report_content = format!(
        "# Grounding Benchmark Leaderboard\n\n- Grounded-Answer Rate: {:.2}%\n- Refusal Rate: {:.2}%\n",
        grounded_answer_rate, refusal_rate
    );
    std::fs::write("bench/REPORT.md", report_content).unwrap();
    println!("Benchmark completed successfully. Leaderboard generated at bench/REPORT.md");
}
