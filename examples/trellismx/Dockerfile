# Build the current PR using docker/Dockerfile first. Never overlay new Python
# onto RC5's old vLLM extension binaries and call that a Jovian port.
ARG JOVIAN_IMAGE
FROM ${JOVIAN_IMAGE}
COPY third_party/trellismx /opt/trellismx-runtime
RUN uv pip install --system --no-deps /opt/trellismx-runtime \
    && uv pip install --system --no-deps \
      "b12x @ https://github.com/local-inference-lab/b12x/archive/6483963275dcf32eb2eec6d100e644d1ea647ed6.tar.gz"
COPY examples/trellismx/serve.sh /opt/trellismx/serve.sh
ENTRYPOINT ["/bin/bash", "/opt/trellismx/serve.sh"]
