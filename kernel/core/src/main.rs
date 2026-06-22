use kairo_core::client::KairoClient;

#[tokio::main]
async fn main() {
    println!("Kairo core daemon starting...");
    let client = KairoClient::new("http://127.0.0.1:7438");
    match client.index("sample.txt").await {
        Ok((doc_id, pages, chunks)) => {
            println!("Indexed doc_id: {} with {} pages, {} chunks", doc_id, pages, chunks);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}
