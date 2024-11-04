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

Python version: 3.11.6

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

## Raspberry Pi 5

Python Version: 3.13.0

| Method    | Format     | Compression   | Progressive   |   Total Time (s) |
|:----------|:-----------|:--------------|:--------------|-----------------:|
| HTTP      | raw_bytes  | None          | N/A           |        0.0286617 |
| gRPC      | raw_bytes  | None          | N/A           |        0.0345874 |
| HTTP      | avif       | 10            | N/A           |        0.0435562 |
| gRPC      | jpeg       | 10            | False         |        0.043967  |
| gRPC      | avif       | 10            | N/A           |        0.0450304 |
| HTTP      | jpeg       | 10            | False         |        0.0465817 |
| HTTP      | raw_base64 | None          | N/A           |        0.0563841 |
| HTTP      | jpeg       | 50            | False         |        0.0640697 |
| gRPC      | raw_base64 | None          | N/A           |        0.0649862 |
| gRPC      | jpeg       | 50            | False         |        0.0649922 |
| gRPC      | avif       | 50            | N/A           |        0.0651193 |
| HTTP      | avif       | 50            | N/A           |        0.0680673 |
| gRPC      | jpeg       | 85            | False         |        0.0790389 |
| HTTP      | jpeg       | 85            | False         |        0.0822692 |
| gRPC      | avif       | 85            | N/A           |        0.0822933 |
| HTTP      | avif       | 85            | N/A           |        0.0826378 |
| WebSocket | jpeg       | 10            | False         |        0.139501  |
| WebSocket | avif       | 10            | N/A           |        0.140209  |
| gRPC      | jpeg       | 10            | True          |        0.190608  |
| HTTP      | jpeg       | 10            | True          |        0.200155  |
| WebSocket | jpeg       | 50            | False         |        0.244813  |
| WebSocket | avif       | 50            | N/A           |        0.249706  |
| gRPC      | png        | 0             | N/A           |        0.255159  |
| HTTP      | png        | 0             | N/A           |        0.255222  |
| WebSocket | jpeg       | 10            | True          |        0.259675  |
| WebSocket | jpeg       | 85            | False         |        0.355558  |
| WebSocket | avif       | 85            | N/A           |        0.388203  |
| gRPC      | jpeg       | 50            | True          |        0.42363   |
| HTTP      | jpeg       | 50            | True          |        0.432249  |
| WebSocket | raw_bytes  | None          | N/A           |        0.445249  |
| gRPC      | png        | 9             | N/A           |        0.528666  |
| HTTP      | png        | 5             | N/A           |        0.533904  |
| HTTP      | png        | 9             | N/A           |        0.534008  |
| gRPC      | png        | 5             | N/A           |        0.53494   |
| WebSocket | raw_base64 | None          | N/A           |        0.53941   |
| gRPC      | jpeg       | 85            | True          |        0.582929  |
| HTTP      | jpeg       | 85            | True          |        0.591892  |
| WebSocket | jpeg       | 50            | True          |        0.595756  |
| WebSocket | jpeg       | 85            | True          |        0.841195  |
| WebSocket | png        | 5             | N/A           |        0.932298  |
| WebSocket | png        | 0             | N/A           |        0.937899  |
| WebSocket | png        | 9             | N/A           |        0.940068  |
| gRPC      | webp       | 10            | N/A           |        2.05821   |
| HTTP      | webp       | 10            | N/A           |        2.07509   |
| WebSocket | webp       | 10            | N/A           |        2.19436   |
| gRPC      | webp       | 50            | N/A           |        2.26503   |
| HTTP      | webp       | 50            | N/A           |        2.27698   |
| WebSocket | webp       | 50            | N/A           |        2.46878   |
| gRPC      | webp       | 85            | N/A           |        2.48822   |
| HTTP      | webp       | 85            | N/A           |        2.50719   |
| WebSocket | webp       | 85            | N/A           |        2.77084   |
