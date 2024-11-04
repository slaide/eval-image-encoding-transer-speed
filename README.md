# Overview

evaluate combinations of transfer protocol+image encoding scheme for lowest latency between `image request sent` and `image received and decoded`.

combines these protocols:
1. HTTP/1.1
2. Websockets
3. gRPC

with these image encoding schemes (each tested at 4 stages of compression strength, as applicable)
1. JPEG
2. PNG
3. AVIF
4. WEBP
5. base64 encoded raw bytes
6. raw bytes

# Results

each combination of `protocol`+`image encoding format`+`image encoding format parameters` is repeated multiple (3) times, and the median reported. the results are sorted by total time below.

total time is `request processing time`+`response decoding time`.

## Apple Silicon M1 Max

| Method    | Format     | Compression   | Progressive   |   Total Time (s) |
|:----------|:-----------|:--------------|:--------------|-----------------:|
| HTTP      | raw_bytes  | None          | N/A           |        0.0135958 |
| gRPC      | raw_bytes  | None          | N/A           |        0.0203631 |
| HTTP      | raw_base64 | None          | N/A           |        0.029645  |
| gRPC      | raw_base64 | None          | N/A           |        0.0436978 |
| HTTP      | avif       | 10            | N/A           |        0.046658  |
| HTTP      | jpeg       | 10            | False         |        0.0475841 |
| gRPC      | jpeg       | 10            | False         |        0.0479763 |
| gRPC      | avif       | 10            | N/A           |        0.0486119 |
| HTTP      | png        | 0             | N/A           |        0.0586262 |
| HTTP      | jpeg       | 85            | False         |        0.0586321 |
| HTTP      | avif       | 85            | N/A           |        0.059175  |
| HTTP      | avif       | 50            | N/A           |        0.0628788 |
| HTTP      | jpeg       | 50            | False         |        0.0629382 |
| gRPC      | jpeg       | 50            | False         |        0.063694  |
| gRPC      | jpeg       | 85            | False         |        0.064616  |
| gRPC      | png        | 0             | N/A           |        0.065228  |
| gRPC      | avif       | 85            | N/A           |        0.0656829 |
| gRPC      | avif       | 50            | N/A           |        0.066402  |
| WebSocket | jpeg       | 10            | False         |        0.11115   |
| WebSocket | avif       | 10            | N/A           |        0.115565  |
| HTTP      | jpeg       | 10            | True          |        0.144903  |
| gRPC      | jpeg       | 10            | True          |        0.145104  |
| WebSocket | jpeg       | 50            | False         |        0.18457   |
| WebSocket | jpeg       | 10            | True          |        0.187602  |
| WebSocket | avif       | 50            | N/A           |        0.188159  |
| WebSocket | jpeg       | 85            | False         |        0.238711  |
| WebSocket | avif       | 85            | N/A           |        0.253808  |
| HTTP      | jpeg       | 50            | True          |        0.287312  |
| gRPC      | jpeg       | 50            | True          |        0.290133  |
| WebSocket | raw_bytes  | None          | N/A           |        0.31914   |
| HTTP      | jpeg       | 85            | True          |        0.359362  |
| gRPC      | jpeg       | 85            | True          |        0.364041  |
| WebSocket | raw_base64 | None          | N/A           |        0.377176  |
| WebSocket | jpeg       | 50            | True          |        0.399846  |
| HTTP      | png        | 5             | N/A           |        0.436638  |
| HTTP      | png        | 9             | N/A           |        0.43741   |
| gRPC      | png        | 9             | N/A           |        0.445214  |
| gRPC      | png        | 5             | N/A           |        0.445364  |
| WebSocket | jpeg       | 85            | True          |        0.535965  |
| WebSocket | png        | 9             | N/A           |        0.722604  |
| WebSocket | png        | 5             | N/A           |        0.727745  |
| WebSocket | png        | 0             | N/A           |        0.73206   |
| gRPC      | webp       | 10            | N/A           |        0.993106  |
| HTTP      | webp       | 10            | N/A           |        1.01265   |
| WebSocket | webp       | 10            | N/A           |        1.07609   |
| HTTP      | webp       | 50            | N/A           |        1.13475   |
| gRPC      | webp       | 50            | N/A           |        1.13757   |
| WebSocket | webp       | 50            | N/A           |        1.27676   |
| HTTP      | webp       | 85            | N/A           |        1.28031   |
| gRPC      | webp       | 85            | N/A           |        1.28084   |
| WebSocket | webp       | 85            | N/A           |        1.44115   |
