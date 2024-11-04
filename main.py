#!/usr/bin/env python3

# requires: numpy Pillow pandas requests websockets grpcio grpcio-tools grpclib protobuf tqdm

import os
import sys
import io
import time
import numpy as np
import asyncio
import pandas as pd
import requests
import grpc
import base64
import hashlib
import json
import websockets
from PIL import Image
from grpc_tools import protoc
from concurrent import futures
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
from threading import Thread
from requests.exceptions import RequestException
from grpc import RpcError
from tqdm import tqdm

# import done, set up pre-computed values

NUM_TEST_REPEATS=3

images=[
    np.random.randint(0, 256, (3000, 3000), dtype=np.uint8)
    for _
    in range(NUM_TEST_REPEATS)
]

# define, compile and import gRPC protobuf spec

# Define Protocol Buffer Specification as a String
PROTO_SPEC = """
syntax = "proto3";

package image;

service ImageService {
  rpc GetImage (ImageRequest) returns (ImageResponse);
}

message ImageRequest {
  string format = 1;           // "jpeg", "png", "webp", etc.
  int32 quality = 2;           // Quality for JPEG/WebP/AVIF (1-100)
  int32 compress_level = 3;    // Compression level for PNG (0-9)
  bool progressive = 4;        // Progressive encoding for JPEG
}

message ImageResponse {
  bytes image_data = 1;        // Raw image bytes
}
"""

# Write the proto spec to a file
PROTO_FILE = "image_service.proto"
with open(PROTO_FILE, "w") as f:
    f.write(PROTO_SPEC)

# Compile the proto file
def compile_proto():
    protoc.main((
        "",
        f"-I.",
        f"--python_out=.",
        f"--grpc_python_out=.",
        PROTO_FILE,
    ))

compile_proto()

# Import the generated protobuf files
import image_service_pb2
import image_service_pb2_grpc

# define server handlers

def img_as_iobuf(img_format,quality=None,compress_level=None,progressive=None,index:int=0)->io.BytesIO|None:
    img_data = images[0]
    img = Image.fromarray(img_data)
    buffer = io.BytesIO()
    try:
        if img_format == "jpeg":
            img.save(buffer, format="JPEG", quality=quality, progressive=progressive)
        elif img_format == "png":
            img.save(buffer, format="PNG", compress_level=compress_level)
        elif img_format == "webp":
            img.save(buffer, format="WEBP", quality=quality)
        elif img_format == "avif":
            img.save(buffer, format="JPEG", quality=quality)
        elif img_format == "raw_bytes":
            buffer.write(img.tobytes())
        elif img_format == "raw_base64":
            buffer.write(base64.b64encode(img.tobytes()))
        else:
            return None

        buffer.seek(0)

    except Exception as e:
        print(f"error ignored - {e}")
        return None

    return buffer

# gRPC Service Implementation
class ImageService(image_service_pb2_grpc.ImageServiceServicer):
    def GetImage(self, request, context):
        buffer = img_as_iobuf(request.format,request.quality,request.compress_level,request.progressive)
        if buffer is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid format specified.")
            return image_service_pb2.ImageResponse()

        return image_service_pb2.ImageResponse(image_data=buffer.getvalue())

# Run the gRPC server
def start_grpc_server():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )
    image_service_pb2_grpc.add_ImageServiceServicer_to_server(ImageService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC server running on port 50051...")
    return server

# HTTP Server Handler with Suppressed Logging
class ImageRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # Override to suppress log output

    def do_GET(self):
        url = urlparse(self.path)

        fmt = url.path.split('/')[1]
        query_params = parse_qs(url.query)

        quality = int(query_params.get("quality", [85])[0]) 
        compress_level = int(query_params.get("compress_level", [6])[0])  
        progressive = query_params.get("progressive", ["false"])[0].lower() == "true"

        buffer = img_as_iobuf(fmt,quality,compress_level,progressive)
        if buffer is None:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid format.")
            return

        self.send_response(200)
        self.end_headers()
        self.wfile.write(buffer.getvalue())

def start_http_server():
    server = TCPServer(("localhost", 8000), ImageRequestHandler)
    server.serve_forever()

# WebSocket Server with Parameter Handling and ACK Verification
async def websocket_handler(websocket, path):
    async for message in websocket:
        # Parse message from the client (expects JSON format)
        try:
            params = json.loads(message)

            fmt = params.get("format", "jpeg")
            quality = int(params.get("quality") or 85)
            compress_level = int(params.get("compress_level") or 6)
            progressive = params.get("progressive", False)
        except (json.JSONDecodeError, ValueError) as e:
            await websocket.send("ERROR: Invalid parameters.")
            continue

        buffer = img_as_iobuf(fmt,quality,compress_level,progressive)
        if buffer is None:
            await websocket.send(f"ERROR: Image processing failed")
            continue

        # Encode as base64 if needed for raw_base64, else use binary
        data = buffer.getvalue()
        chunk_size = 64 * 1024  # 64 KB chunks

        # Calculate checksum and send START message with data size and checksum
        checksum = hashlib.sha256(data).hexdigest()
        await websocket.send(f"START:{len(data)}:{checksum}")

        # Send data in chunks, waiting for 'ACK' after each chunk
        for i in range(0, len(data), chunk_size):
            await websocket.send(data[i:i + chunk_size])
            ack = await websocket.recv()
            if ack != "ACK":
                await websocket.send("ERROR: Missing ACK, transmission aborted.")
                break

        # Signal the end of transmission
        await websocket.send("END")

# Increase WebSocket buffer size for server
async def start_websocket_server():
    server = await websockets.serve(websocket_handler, "localhost", 8001, max_size=100 * 1024 * 1024)
    await server.wait_closed()

# Use a new function to run the asyncio event loop in a thread
def run_websocket_server():
    asyncio.run(start_websocket_server())


# gRPC Client with Increased Message Limits
def request_images_grpc(format="jpeg", quality=85, compress_level=6, progressive=False):
    try:
        options = [
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
        channel = grpc.insecure_channel("localhost:50051", options=options)
        stub = image_service_pb2_grpc.ImageServiceStub(channel)
        request = image_service_pb2.ImageRequest(
            format=format, quality=quality, compress_level=compress_level, progressive=progressive
        )
        start_time = time.time()
        response = stub.GetImage(request)
        elapsed_time = time.time() - start_time
        
        if format == "raw_bytes":
            img_array = np.frombuffer(response.image_data, dtype=np.uint8).reshape((3000, 3000))
        elif format == "raw_base64":
            raw_bytes = base64.b64decode(response.image_data)
            img_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((3000, 3000))
        else:
            img = Image.open(io.BytesIO(response.image_data))
            img_array = np.array(img)

        return elapsed_time, img_array
    except RpcError as e:
        print(f"gRPC error: {e}")
        return None, None

# HTTP Client with Raw and Base64 Handling
def request_images_http(fmt, quality=None, compress_level=None, progressive=False):
    url = f"http://localhost:8000/{fmt}"
    params = {}
    if quality is not None:
        params["quality"] = quality
    if compress_level is not None:
        params["compress_level"] = compress_level
    if progressive:
        params["progressive"] = "true"
        
    start_time = time.time()
    response = requests.get(url, params=params, timeout=5)
    elapsed_time = time.time() - start_time

    if fmt == "raw_bytes":
        img_array = np.frombuffer(response.content, dtype=np.uint8).reshape((3000, 3000))
    elif fmt == "raw_base64":
        raw_bytes = base64.b64decode(response.content)
        img_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((3000, 3000))
    else:
        img = Image.open(io.BytesIO(response.content))
        img_array = np.array(img)

    return elapsed_time, img_array

# WebSocket Client with Raw and Base64 Handling and Data Size Verification
async def request_images_ws(fmt, quality=None, compress_level=None, progressive=False):
    uri = "ws://localhost:8001"
    async with websockets.connect(uri, timeout=5, max_size=100 * 1024 * 1024) as websocket:
        # Create parameters JSON
        params = {
            "format": fmt,
            "quality": quality,
            "compress_level": compress_level,
            "progressive": progressive,
        }

        start_time = time.time()

        await websocket.send(json.dumps(params))  # Send as JSON string

        full_data = bytearray()
        expected_size = None
        receiving = False

        while True:
            chunk = await websocket.recv()
            if isinstance(chunk, str) and (chunk.startswith("START") or chunk.startswith("END") or chunk.startswith("ERROR")):
            
                if chunk.startswith("START"):
                    receiving = True
                    full_data = bytearray()
                    expected_size = int(chunk.split(":")[1])
                elif chunk == "END":
                    receiving = False
                    break
                elif chunk.startswith("ERROR"):
                    raise Exception(chunk)
                else:
                    raise Exception(f"unexpected chunk: {chunk}")

            elif receiving:
                if isinstance(chunk, str):
                    chunk=chunk.encode()
                full_data.extend(chunk)

            # Acknowledge receipt of each chunk
            await websocket.send("ACK")

        elapsed_time = time.time() - start_time

        # Decode data if base64-encoded, check size
        if fmt == "raw_base64":
            decoded_data = base64.b64decode(full_data)
        else:
            decoded_data = full_data

        # Validate size and reshape for raw formats
        if expected_size is not None and len(full_data) != expected_size:
            raise ValueError(f"Received data size {len(full_data)} does not match expected size {expected_size}.")

        if fmt in ["raw_bytes", "raw_base64"]:
            img_array = np.frombuffer(decoded_data, dtype=np.uint8).reshape((3000, 3000))
        else:
            img = Image.open(io.BytesIO(decoded_data))
            img_array = np.array(img)

        return elapsed_time, img_array

# Run Tests with Raw and Base64 Handling and Progress Bars
results = []
def run_tests(num_repeats=3):
    formats = {
        "jpeg": {"quality_levels": [10, 50, 85], "progressive": [False, True]},
        "png": {"compress_levels": [0, 5, 9]},
        "webp": {"quality_levels": [10, 50, 85]},
        "avif": {"quality_levels": [10, 50, 85]},
        "raw_bytes": {},
        "raw_base64": {}
    }

    for fmt, options in tqdm(formats.items(), desc="Formats"):
        quality_levels = options.get("quality_levels", [None])
        compress_levels = options.get("compress_levels", [None])
        progressive_options = options.get("progressive", [None])

        for quality in tqdm(quality_levels, desc=f"{fmt} Quality Levels", leave=False):
            for compress_level in tqdm(compress_levels, desc=f"{fmt} Compression Levels", leave=False):
                for progressive in tqdm(progressive_options, desc=f"{fmt} Progressive Options", leave=False):
                    for method in tqdm(["HTTP", "WebSocket", "gRPC"], desc="Methods", leave=False):
                        for _ in tqdm(range(num_repeats), desc="Repeats", leave=False):
                            try:
                                if method == "HTTP":
                                    elapsed_time, img_array = request_images_http(fmt, quality, compress_level, progressive)
                                elif method == "WebSocket":
                                    elapsed_time, img_array = asyncio.run(request_images_ws(fmt, quality, compress_level, progressive))
                                elif method == "gRPC":
                                    elapsed_time, img_array = request_images_grpc(fmt, quality, compress_level, progressive)
                                
                                results.append({
                                    "Method": method,
                                    "Format": fmt,
                                    "Compression": f"{quality or compress_level}",
                                    "Progressive": progressive,
                                    "Total Time (s)": elapsed_time
                                })

                            except Exception as e:
                                raise e
                                print(f"{method} error for format {fmt} with quality {quality}: {e}")
                                results.append({
                                    "Method": method,
                                    "Format": fmt,
                                    "Compression": f"{quality or compress_level}",
                                    "Progressive": progressive,
                                    "Total Time (s)": None
                                })

    # Display results ordered by Total Time
    df = pd.DataFrame(results)
    df = df.sort_values(by="Total Time (s)")
    print("\nTest Results:")
    print(df.to_string(index=False))

# Display summarized results with median total time
def summarize_results(results):
    df = pd.DataFrame(results)

    # replace "None" values to include them in groupby
    df.fillna("N/A", inplace=True)

    # Calculate the median total time for each group
    median_df = df.groupby(["Method", "Format", "Compression", "Progressive"], as_index=False)["Total Time (s)"].median()
    median_df = median_df.sort_values(by="Total Time (s)")
    print("\nMedian Test Results:")
    print(median_df.to_string(index=False))

# Run Tests and Display Summarized Results
if __name__ == "__main__":
    compile_proto()
    grpc_server = start_grpc_server()
    Thread(target=start_http_server, daemon=True).start()
    Thread(target=run_websocket_server, daemon=True).start()
    time.sleep(2)  # Give servers a moment to start
    run_tests(num_repeats=NUM_TEST_REPEATS)
    grpc_server.stop(0)
    summarize_results(results)

