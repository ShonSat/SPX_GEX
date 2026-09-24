use serde::Deserialize;
use std::collections::HashMap;

const YOUR_API_KEY: &str = "";
const BASE: &str = "https://lab.flashalpha.com";
const API_KEY: &str = "YOUR_API_KEY";

// 1. Define the structural contracts mapping to the FlashAlpha JSON responses
#[derive(Deserialize, Debug)]
struct ReferenceData {
    underlying_price: f64,
    net_gex: f64,
    gamma_flip: f64,
}

#[derive(Deserialize, Debug)]
struct ExpirationsResponse {
    expirations: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct OptionQuote {
    #[serde(rename = "type")]
    option_type: String, // "C" or "P"
    gamma: Option<f64>,
    open_interest: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2. Initialize a reusable asynchronous HTTP client
    let client = reqwest::Client::new();

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("X-Api-Key", reqwest::header::HeaderValue::from_static(API_KEY));

    // 3. Fetch reference data and pull the current spot underlying price
    let ref_url = format!("{}/v1/exposure/gex/SPY", BASE);
    let reference: ReferenceData = client.get(&ref_url)
        .headers(headers.clone())
        .send()
        .await?
        .json()
        .await?;

    let s = reference.underlying_price;

    // 4. Ingest the total array of expiration dates available
    let expiries_url = format!("{}/v1/options/SPY", BASE);
    let expiries_res: ExpirationsResponse = client.get(&expiries_url)
        .headers(headers.clone())
        .send()
        .await?
        .json()
        .await?;

    let mut net = 0.0;

    // 5. Loop over each expiration date to collect the granular contract quotes
    for exp in expiries_res.expirations {
        let quote_url = format!("{}/optionquote/SPY", BASE);

        // Pass query parameters safely matching Python params={"expiry": exp}
        let quotes: Vec<OptionQuote> = client.get(&quote_url)
            .query(&[("expiry", &exp)])
            .headers(headers.clone())
            .send()
            .await?
            .json()
            .await?;

        // 6. Aggregate the total notional GEX matrix
        for c in quotes {
            if let (Some(gamma), Some(oi)) = (c.gamma, c.open_interest) {
                let gex = gamma * oi * 100.0 * s * s * 0.01;

                if c.option_type == "C" {
                    net += gex;
                } else {
                    net -= gex;
                }
            }
        }
    }

    // 7. Render final telemetry benchmarks
    println!("hand-rolled net GEX: ${:.2}B per 1% move", net / 1e9);
    println!(
        "API net GEX:         ${:.2}B, flip {}",
        reference.net_gex / 1e9,
        reference.gamma_flip
    );

    Ok(())
}
