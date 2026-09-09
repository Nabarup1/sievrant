use clap::{Parser, Subcommand};
use reqwest::Client;
use serde_json::json;

#[derive(Parser)]
#[command(name = "sievrant", version, about = "CLI for Sievrant Retrieval Engine")]
struct Cli {
    #[arg(short, long, default_value = "http://localhost:8080")]
    url: String,

    #[arg(short, long, env = "SIEVRANT_API_KEY")]
    token: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Check server health
    Health,
    /// Query a corpus
    Query {
        #[arg(short, long)]
        corpus: String,
        query: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let client = Client::new();

    match &cli.command {
        Commands::Health => {
            let res = client.get(&format!("{}/health", cli.url)).send().await?;
            println!("Status: {}", res.status());
            if res.status().is_success() {
                println!("{}", res.text().await?);
            }
        }
        Commands::Query { corpus, query } => {
            let mut req = client.post(&format!("{}/v1/query", cli.url));
            if let Some(token) = &cli.token {
                req = req.bearer_auth(token);
            }
            
            let res = req.json(&json!({
                "corpus": corpus,
                "query": query,
                "config": {}
            })).send().await?;
            
            println!("Status: {}", res.status());
            println!("{}", serde_json::to_string_pretty(&res.json::<serde_json::Value>().await?)?);
        }
    }

    Ok(())
}
