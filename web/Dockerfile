FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build g++ python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTIRAMISU_BUILD_TESTS=OFF \
    -DTIRAMISU_BUILD_BENCH=OFF \
    -DTIRAMISU_BUILD_EXAMPLES=ON
RUN cmake --build build --parallel --target train_shakespeare

FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/examples/train_shakespeare ./build/examples/train_shakespeare
COPY --from=builder /app/checkpoints ./checkpoints

COPY web/ ./web/

RUN pip3 install --no-cache-dir --break-system-packages \
    fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.8.2

EXPOSE 8000

CMD ["python3", "web/server.py"]